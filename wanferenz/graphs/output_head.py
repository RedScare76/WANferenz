import torch


class CapturedOutput:
    def __init__(self, stage, h):
        self.stage = stage
        self.h = torch.empty_like(h)
        self.graph = None
        self.failed = False

    def _forward(self):
        logits = self.stage.logits_all(self.h, full_logits=False)
        return logits, logits.argmax(dim=-1)

    def _capture(self):
        side = torch.cuda.Stream(device=self.h.device)
        side.wait_stream(torch.cuda.current_stream(self.h.device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream(self.h.device).wait_stream(side)
        torch.cuda.synchronize(self.h.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            self.logits, self.token = self._forward()
        self.graph = graph
        print("[v4] main head graph captured", flush=True)

    def run(self, h):
        with torch.cuda.device(h.device):
            return self._run(h)

    def _run(self, h):
        if self.failed:
            return None
        self.h.copy_(h)
        if self.graph is None:
            try:
                self._capture()
            except RuntimeError as exc:
                torch.cuda.synchronize(self.h.device)
                self.failed = True
                print(
                    f"[v4] main head graph declined: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return None
        self.graph.replay()

        return int(self.token.item())
