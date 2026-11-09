from contextlib import contextmanager
import json

import numpy as np
import pytest

from oppomem.memory.rps import abstract_response_strategy, has_concrete_action_name
from oppomem.prompts.rps import player_prompt
from oppomem.memory import CentroidMemoryBank, MECHANISM, batch_reflection_prompt
from oppomem.memory.evolution import reflect_all, commit, Proposal
from oppomem.types import CentroidCluster


@pytest.mark.parametrize('advice', ['Play Rock.', 'Play Paper.', 'Play Scissors.', '出石头', '出石頭', '出布', '出剪刀', '', '   '])
def test_reject_invalid_advice(advice):
    with pytest.raises(ValueError):
        abstract_response_strategy(advice)


def test_preserve_distinct_advice_and_tom_without_memory():
    first = 'Infer the repeated action from recent history and counter it.'
    second = 'Infer the cycle phase from recent history and counter its next action.'
    assert abstract_response_strategy(first) == first
    assert abstract_response_strategy(second) == second
    assert first != second
    prompt = player_prompt([], horizon=20, tom='A fixed preference.', memory=None)
    assert 'A fixed preference.' in prompt
    assert 'retrieved_memory' not in prompt


def payload():
    return dict(canonical_opponent_strategy='A fixed preference.', previous_response_strategy=None,
                support_segments=1, evidence=[], guidance_shown=[], injection_contrast={},
                reflection_phase='retrieved_update')


class Pool:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    @contextmanager
    def lease(self):
        yield self

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return {'response_strategy': next(self.responses)}


def test_reflection_corrects_invalid_advice_twice(tmp_path):
    good = 'Infer the next action from recent switching and counter it.'
    pool = Pool(['Play Rock.', '', good])
    result = reflect_all({'cluster': payload()}, pool, partial_path=tmp_path / 'partial.json', abstract_actions=True)
    assert result['cluster']['response_strategy'] == good
    assert len(pool.calls) == 3
    assert 'RETRIEVED UPDATE' in pool.calls[0]['prompt']


def test_reflection_failure_is_not_published(tmp_path):
    path = tmp_path / 'partial.json'
    pool = Pool(['Play Rock.'] * 3)
    with pytest.raises(ValueError):
        reflect_all({'cluster': payload()}, pool, partial_path=path, abstract_actions=True)
    assert not path.exists()
    assert len(pool.calls) == 3
    assert (tmp_path / 'partial.cluster.attempts.json').exists()


def test_bank_boundary_and_response_only_update(tmp_path):
    bank = CentroidMemoryBank(tmp_path / 'bank.json')
    bank.metadata = {'mechanism': MECHANISM, 'abstract_rps_memory': True}
    cluster = CentroidCluster(cluster_id='a', opponent_strategy='Always Rock.',
                              response_strategy='Infer and counter the repeated action.',
                              centroid=[1., 0.], centroid_size=4,
                              prototype_strategy='Repeated Paper.',
                              members=[{'opponent_strategy': '一直出剪刀', 'embedding': [1., 0.]}])
    bank.clusters['a'] = cluster
    bank.save()
    assert not has_concrete_action_name(json.dumps(json.loads(bank.path.read_text(encoding="utf-8"))['clusters'], ensure_ascii=False))
    advice = 'Infer the next switch from completed rounds and counter the predicted action.'
    proposal = Proposal('a', cluster.opponent_strategy, cluster.response_strategy,
                        np.array(cluster.centroid), 4, cluster)
    operations = commit(bank, {'a': proposal}, {'a': {'response_strategy': advice}}, None,
                        metadata={}, abstract_rps=True)
    assert operations[0]['operation'] == 'response_updated'
    assert cluster.centroid_size == 4
    assert cluster.centroid == [1., 0.]
    assert CentroidMemoryBank(bank.path).load().clusters['a'].response_strategy == advice
    previous = bank.path.read_bytes()
    cluster.response_strategy = 'Play Paper.'
    with pytest.raises(ValueError):
        bank.save()
    assert bank.path.read_bytes() == previous
