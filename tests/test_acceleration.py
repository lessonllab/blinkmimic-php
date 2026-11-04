import copy
import unittest

import torch

from supermimic_parkour.acceleration import configure_adam, finite_adam_step, GraphedUpdate
from supermimic_parkour.buffer import ReplayBuffer
from supermimic_parkour.networks import Policy


class AccelerationTests(unittest.TestCase):
    def test_batched_encoding_preserves_outputs_and_encoder_gradients(self):
        torch.manual_seed(7)
        sequential = Policy(160, 680, hist=2, hidden=(16, 8))
        batched = copy.deepcopy(sequential)
        goals, observations = torch.randn(5, 3, 680), torch.randn(5, 3, 2, 160)
        expected = torch.stack([sequential(observations[t], goals[t]) for t in range(5)])
        latent = batched.encode(goals.flatten(0, 1)).view(5, 3, -1)
        actual = torch.stack([batched.decode(observations[t], latent[t]) for t in range(5)])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for a, b in zip(sequential.parameters(), batched.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=2e-5)

    def test_certified_replay_still_rejects_numerical_failures(self):
        buffer = ReplayBuffer(3, 4, 'cpu', track_validity=True)

        def add(valid):
            row = {k: torch.zeros_like(v[0]) for k, v in buffer.data.items()}
            row['valid'][:] = torch.tensor(valid)
            row['s'][~row['valid']] = 99
            buffer.add(row, all_valid=all(valid))

        add([True, True, True, True])
        add([False, True, True, True])
        add([True, True, True, True])
        for _ in range(5):
            starts, _ = buffer.sample_starts(128, 1)
            self.assertFalse(bool((starts['s'] == 99).any()))
        add([True, True, True, True])
        add([True, True, True, True])
        add([True, True, True, True])
        self.assertGreaterEqual(buffer.add_count, buffer.invalid_until)
        self.assertTrue(bool(buffer.sample_windows(128, 2)['valid'].all()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_nonfinite_gate_preserves_parameters_moments_and_step(self):
        p = torch.nn.Parameter(torch.tensor([1., -2.], device='cuda'))
        optimizer = torch.optim.Adam([p], lr=0.01, fused=True)
        skipped = torch.zeros((), device='cuda', dtype=torch.long)
        p.grad = torch.tensor([0.3, -0.4], device='cuda')
        finite_adam_step(optimizer, torch.linalg.vector_norm(p.grad), skipped)
        value = p.detach().clone()
        state = {k: v.clone() for k, v in optimizer.state[p].items()}
        for bad in (float('nan'), float('inf')):
            p.grad = torch.full_like(p, bad)
            finite_adam_step(optimizer, torch.tensor(bad, device='cuda'), skipped)
            torch.testing.assert_close(p, value, atol=0, rtol=0)
            for key, tensor in state.items():
                torch.testing.assert_close(optimizer.state[p][key], tensor, atol=0, rtol=0)
        self.assertEqual(skipped.item(), 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_legacy_adam_resume_matches_next_update(self):
        p = torch.nn.Parameter(torch.tensor([1., -2.], device='cuda'))
        old = torch.optim.Adam([p], lr=0.01)
        p.grad = torch.tensor([0.3, -0.4], device='cuda')
        old.step()
        resumed = torch.nn.Parameter(p.detach().clone())
        fast = torch.optim.Adam([resumed], lr=0.01, fused=True)
        fast.load_state_dict(copy.deepcopy(old.state_dict()))
        configure_adam(fast, fused=True)
        self.assertTrue(fast.param_groups[0]['fused'])
        self.assertEqual(fast.state[resumed]['step'].device.type, 'cuda')
        p.grad = torch.tensor([0.2, 0.1], device='cuda')
        resumed.grad = p.grad.clone()
        old.step()
        finite_adam_step(fast, torch.linalg.vector_norm(resumed.grad), torch.zeros((), device='cuda', dtype=torch.long))
        torch.testing.assert_close(p, resumed, atol=1e-7, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_graph_capture_does_not_add_updates_and_lr_stays_live(self):
        torch.manual_seed(11)
        regular = torch.nn.Linear(4, 2, device='cuda')
        graphed = copy.deepcopy(regular)
        optimizers = [torch.optim.Adam(m.parameters(), lr=0.01, fused=True) for m in (regular, graphed)]
        for optimizer in optimizers:
            configure_adam(optimizer, fused=True, capturable=True)
        counters = [torch.zeros((), dtype=torch.long, device='cuda') for _ in optimizers]

        def update(module, optimizer, counter, inputs):
            optimizer.zero_grad(set_to_none=True)
            loss = (module(inputs['x']) - inputs['y']).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            finite_adam_step(optimizer, norm, counter)
            return (loss.detach(),)

        inputs = {'x': torch.randn(8, 4, device='cuda'), 'y': torch.randn(8, 2, device='cuda')}
        for m, o, c in zip((regular, graphed), optimizers, counters):
            update(m, o, c, inputs)  # initialize identical optimizer states
        wrapped = GraphedUpdate(lambda data: update(graphed, optimizers[1], counters[1], data), graphed, optimizers[1], (counters[1],))
        saved_metric = None
        for lr in (0.01, 0.003, 0.001):
            for optimizer in optimizers:
                optimizer.param_groups[0]['lr'].fill_(lr)
            expected = update(regular, optimizers[0], counters[0], inputs)
            actual = wrapped(inputs)
            torch.testing.assert_close(actual[0], expected[0])
            for p, q in zip(regular.parameters(), graphed.parameters()):
                torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)
                for key in optimizers[0].state[p]:
                    torch.testing.assert_close(optimizers[0].state[p][key], optimizers[1].state[q][key])
            if saved_metric is None:
                saved_metric = (actual[0], actual[0].clone())
        torch.testing.assert_close(*saved_metric, atol=0, rtol=0)
        before = [p.detach().clone() for p in graphed.parameters()]
        bad = {**inputs, 'y': torch.full_like(inputs['y'], float('nan'))}
        wrapped(bad)
        for p, old in zip(graphed.parameters(), before):
            torch.testing.assert_close(p, old, atol=0, rtol=0)
            self.assertEqual(optimizers[1].state[p]['step'].item(), 4)
        self.assertEqual(counters[1].item(), 1)


if __name__ == '__main__':
    unittest.main()
