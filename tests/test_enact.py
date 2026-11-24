from selectflow import scenario_greece as G
from selectflow.agent.claude import ClaudeConfig, tool_name
from selectflow.agent.enact import Enactor, Episode, MockEnvironment


def test_enact_loop_with_scripted_backend():
    fed = G.build()
    env = MockEnvironment(fed, {"tax_registration", "fire_certificate"})
    ep = Episode("system", ["muni.attestation.get", "notify.new.submit"], env,
                 [{"role": "user", "content": "notify the opening"}])
    script = [["muni.attestation.get"], ["notify.new.submit"], []]

    def backend(params_list):
        calls = script.pop(0)
        return [{"content": [{"type": "tool_use", "id": f"u{i}", "name": tool_name(t), "input": {}}
                             for i, t in enumerate(calls)] or [{"type": "text", "text": "done"}]}]

    Enactor(fed, ClaudeConfig(cache=False)).run([ep], backend)
    assert "receipt" in env.held and ep.done
