from e3a.adapters.sam3_video import SAM31VideoTeacher


class FakePredictor:
    def __init__(self):
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        return {"ok": True, "request": request}

    def handle_stream_request(self, request):
        self.requests.append(request)
        yield {"frame_index": 0, "outputs": {}}
        yield {"frame_index": 1, "outputs": {}}


def test_sam3_teacher_uses_official_request_contract():
    predictor = FakePredictor()
    teacher = SAM31VideoTeacher(predictor=predictor)

    teacher.start_session("/tmp/video")
    teacher.add_text_prompt("session-1", 0, "a mug")
    frames = list(teacher.propagate("session-1", direction="forward"))
    teacher.close_session("session-1")

    assert [item["type"] for item in predictor.requests] == [
        "start_session",
        "add_prompt",
        "propagate_in_video",
        "close_session",
    ]
    assert predictor.requests[1]["text"] == "a mug"
    assert len(frames) == 2
