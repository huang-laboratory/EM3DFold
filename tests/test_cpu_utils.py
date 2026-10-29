import os
import unittest
from unittest.mock import patch

from em3dfold.utils.cpu_utils import (
    MAX_CPU_THREADS_ENV,
    _THREAD_ENV_VARS,
    configure_cpu_resources,
    configure_loaded_thread_pools,
)


class CpuUtilsTests(unittest.TestCase):
    def test_default_caps_environment_and_affinity_at_four(self):
        clean_environment = {
            key: value
            for key, value in os.environ.items()
            if key not in (*_THREAD_ENV_VARS, MAX_CPU_THREADS_ENV)
        }
        with patch.dict(os.environ, clean_environment, clear=True):
            with patch("os.sched_getaffinity", return_value=set(range(16))):
                with patch("os.sched_setaffinity") as set_affinity:
                    config = configure_cpu_resources()
                    thread_environment = {
                        variable_name: os.environ[variable_name]
                        for variable_name in _THREAD_ENV_VARS
                    }

        self.assertEqual(config.max_threads, 4)
        self.assertEqual(config.affinity_after, (0, 1, 2, 3))
        set_affinity.assert_called_once_with(0, (0, 1, 2, 3))
        for variable_name in _THREAD_ENV_VARS:
            self.assertEqual(thread_environment[variable_name], "4")

    def test_respects_smaller_existing_limits_and_scheduler_affinity(self):
        environment = {
            MAX_CPU_THREADS_ENV: "4",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "8",
        }
        with patch.dict(os.environ, environment, clear=True):
            with patch("os.sched_getaffinity", return_value={20, 21}):
                with patch("os.sched_setaffinity") as set_affinity:
                    config = configure_cpu_resources()
                    omp_threads = os.environ["OMP_NUM_THREADS"]
                    openblas_threads = os.environ["OPENBLAS_NUM_THREADS"]

        self.assertEqual(config.affinity_after, (20, 21))
        self.assertEqual(omp_threads, "2")
        self.assertEqual(openblas_threads, "4")
        set_affinity.assert_not_called()

    def test_environment_can_request_a_lower_limit(self):
        with patch.dict(os.environ, {MAX_CPU_THREADS_ENV: "1"}, clear=True):
            with patch("os.sched_getaffinity", return_value={4, 5, 6}):
                with patch("os.sched_setaffinity") as set_affinity:
                    config = configure_cpu_resources()

        self.assertEqual(config.max_threads, 1)
        self.assertEqual(config.affinity_after, (4,))
        set_affinity.assert_called_once_with(0, (4,))

    def test_environment_cannot_raise_the_hard_limit(self):
        with patch.dict(os.environ, {MAX_CPU_THREADS_ENV: "32"}, clear=True):
            with patch("os.sched_getaffinity", return_value=set(range(16))):
                with patch("os.sched_setaffinity") as set_affinity:
                    config = configure_cpu_resources()

        self.assertEqual(config.max_threads, 4)
        self.assertEqual(config.affinity_after, (0, 1, 2, 3))
        set_affinity.assert_called_once_with(0, (0, 1, 2, 3))

    def test_invalid_limit_is_rejected(self):
        with patch.dict(os.environ, {MAX_CPU_THREADS_ENV: "many"}, clear=True):
            with self.assertRaisesRegex(ValueError, MAX_CPU_THREADS_ENV):
                configure_cpu_resources()

    def test_only_already_loaded_thread_pools_are_configured(self):
        torch = unittest.mock.Mock()
        numba = unittest.mock.Mock()
        numba.get_num_threads.return_value = 12
        with patch.dict(
            "sys.modules",
            {"torch": torch, "numba": numba},
            clear=False,
        ):
            configure_loaded_thread_pools(4)

        torch.set_num_threads.assert_called_once_with(4)
        torch.set_num_interop_threads.assert_called_once_with(4)
        numba.set_num_threads.assert_called_once_with(4)


if __name__ == "__main__":
    unittest.main()
