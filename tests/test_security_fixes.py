"""Tests for the interactive nodes' temp files, wait loop and selection route.

Run with ComfyUI's Python from the toolkit folder:
    python -m unittest discover -s tests -v
"""
import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest

TOOLKIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_DIR = os.path.dirname(os.path.dirname(TOOLKIT_DIR))
sys.path.insert(0, COMFY_DIR)
sys.path.insert(0, TOOLKIT_DIR)

# Answers the browser would send, keyed by node id. The fake server posts them
# from a thread after the node starts waiting, like the frontend does.
REPLIES = {}


class _Routes:
    def get(self, path):
        return lambda f: f

    def post(self, path):
        return lambda f: f


class _FakePromptServer:
    routes = _Routes()

    def send_sync(self, event, data):
        node_id = data["node_id"]
        if node_id in REPLIES:
            def answer():
                time.sleep(0.2)
                sg.WAITING_SELECTIONS[node_id] = REPLIES.pop(node_id)
            threading.Thread(target=answer, daemon=True).start()


_server = types.ModuleType("server")
_server.PromptServer = type("PromptServer", (), {"instance": _FakePromptServer()})
sys.modules["server"] = _server

import torch  # noqa: E402
import folder_paths  # noqa: E402
import comfy.model_management as mm  # noqa: E402
from comfy.model_management import InterruptProcessingException  # noqa: E402
import space_gremlin_nodes as sg  # noqa: E402


class _Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class SecurityFixesTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self._orig_get_temp = folder_paths.get_temp_directory
        folder_paths.get_temp_directory = lambda: self.temp_dir
        self.images = torch.rand(10, 32, 48, 3)
        REPLIES.clear()
        sg.WAITING_SELECTIONS.clear()
        sg.WAITING_DATA.clear()
        mm.interrupt_current_processing(False)

    def tearDown(self):
        folder_paths.get_temp_directory = self._orig_get_temp
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        mm.interrupt_current_processing(False)

    def assert_no_waiting_state(self, node_id):
        self.assertNotIn(node_id, sg.WAITING_SELECTIONS)
        self.assertNotIn(node_id, sg.WAITING_DATA)

    def test_temp_filename_strips_path_characters(self):
        name = sg._temp_filename("cmp_orig", r"..\..\..\evil/../x:5", "png")
        self.assertEqual(name, "cmp_orig__________evil____x_5.png")
        self.assertEqual(sg._temp_filename("frame_picker_grid", "12", "jpg"), "frame_picker_grid_12.jpg")

    def test_compare_selector_keeps_files_in_temp_dir(self):
        node_id = r"..\..\x"
        REPLIES[node_id] = "original"
        sg.ImageCompareSelector().compare_and_select(
            [self.images[:1]], [self.images[1:2]], [False], unique_id=[node_id])
        files = os.listdir(self.temp_dir)
        self.assertEqual(sorted(files), ["cmp_enh_______x.png", "cmp_orig_______x.png"])
        self.assert_no_waiting_state(node_id)

    def test_frame_picker_filters_and_caps_indices(self):
        REPLIES["7"] = {"action": "continue",
                        "indices": [3, -1, 99, "2", True, 1.0, 4, 5, 6, 7, 8, 9] + [0] * 100000}
        out, indices = sg.SpaceGremlinFramePicker().pick_frames(
            self.images, min_selection=2, max_selection=4, unique_id="7")
        self.assertEqual(indices, "3,4,5,6")
        self.assertEqual(tuple(out.shape), (4, 32, 48, 3))
        self.assert_no_waiting_state("7")

    def test_frame_picker_falls_back_on_non_list_indices(self):
        REPLIES["8"] = {"action": "continue", "indices": "garbage"}
        out, indices = sg.SpaceGremlinFramePicker().pick_frames(
            self.images, min_selection=2, max_selection=4, unique_id="8")
        self.assertEqual(indices, "0,1")
        self.assertEqual(out.shape[0], 2)

    def test_frame_picker_cancel_interrupts(self):
        REPLIES["11"] = {"action": "cancel", "indices": []}
        with self.assertRaises(InterruptProcessingException):
            sg.SpaceGremlinFramePicker().pick_frames(self.images, unique_id="11")
        self.assert_no_waiting_state("11")

    def test_pause_continue_passes_values_through(self):
        REPLIES["12"] = "continue"
        self.assertEqual(sg.PauseControl().pause_execution(1, 2, unique_id="12"), (1, 2))
        self.assert_no_waiting_state("12")

    def test_interrupt_stops_waiting_node(self):
        def interrupt():
            time.sleep(0.2)
            mm.interrupt_current_processing(True)
        threading.Thread(target=interrupt, daemon=True).start()
        with self.assertRaises(InterruptProcessingException):
            sg.PauseControl().pause_execution(1, 2, unique_id="9")
        self.assert_no_waiting_state("9")
        self.assertFalse(mm.processing_interrupted())

    def test_select_route_rejects_invalid_choice(self):
        sg.WAITING_SELECTIONS["10"] = None
        for bad in (None, 5, ["continue"]):
            resp = asyncio.run(sg.receive_selection(_Request({"node_id": "10", "choice": bad})))
            self.assertEqual(resp.status, 400)
            self.assertIsNone(sg.WAITING_SELECTIONS["10"])

        resp = asyncio.run(sg.receive_selection(_Request({"node_id": "10", "choice": "continue"})))
        self.assertEqual(resp.status, 200)
        self.assertEqual(sg.WAITING_SELECTIONS["10"], "continue")

    def test_select_route_rejects_node_not_waiting(self):
        resp = asyncio.run(sg.receive_selection(_Request({"node_id": "404", "choice": "continue"})))
        self.assertEqual(resp.status, 400)
        self.assertNotIn("404", sg.WAITING_SELECTIONS)


if __name__ == "__main__":
    unittest.main()
