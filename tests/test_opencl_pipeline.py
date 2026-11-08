"""Тесты для OpenCL аппаратного ускорения и graceful fallback на CPU."""
import unittest
import numpy as np
import cv2
from core.pipeline import PipelineWorker


class TestOpenCLPipeline(unittest.TestCase):
    def setUp(self):
        self.worker = PipelineWorker()
        self.task_id = self.worker._task_id

    def test_opencl_initialization(self):
        # Проверяем корректную инициализацию флага доступности OpenCL
        self.assertTrue(hasattr(self.worker, '_has_ocl'))
        self.assertIsInstance(self.worker._has_ocl, bool)
        if cv2.ocl.haveOpenCL():
            self.assertTrue(cv2.ocl.useOpenCL())

    def test_compute_stage2_cpu_and_umat_precision(self):
        # 512x512 тестовое изображение
        np.random.seed(42)
        img = np.random.uniform(0.1, 0.9, (512, 512, 3)).astype(np.float32)

        # CPU расчет
        res_cpu = self.worker._compute_stage2(
            img, 'test_key', sigma=1.5, iterations=3, protect_edge=True,
            task_id=self.task_id, prefer_umat=False
        )
        self.assertIsNotNone(res_cpu)
        self.assertEqual(res_cpu.shape, img.shape)

        # OpenCL UMat расчет
        if self.worker._has_ocl:
            res_umat = self.worker._compute_stage2(
                img, 'test_key', sigma=1.5, iterations=3, protect_edge=True,
                task_id=self.task_id, prefer_umat=True
            )
            self.assertIsNotNone(res_umat)
            self.assertEqual(res_umat.shape, img.shape)

            # Проверка численной точности
            max_diff = np.max(np.abs(res_cpu - res_umat))
            self.assertLess(max_diff, 1e-5, f"OpenCL vs CPU max difference too high: {max_diff}")

    def test_opencl_graceful_fallback_on_error(self):
        # Имитируем сбой OpenCL в _compute_stage2_umat
        img = np.ones((128, 128, 3), dtype=np.float32)

        original_compute_umat = self.worker._compute_stage2_umat

        def failing_compute_umat(*args, **kwargs):
            # Симуляция cv2.error
            raise cv2.error("Simulated OpenCL driver timeout")

        self.worker._compute_stage2_umat = failing_compute_umat
        try:
            # При prefer_umat=True должен произойти graceful fallback на CPU без исключений
            res = self.worker._compute_stage2(
                img, 'test_fallback', sigma=1.0, iterations=2, protect_edge=False,
                task_id=self.task_id, prefer_umat=True
            )
            self.assertIsNotNone(res, "Fallback to CPU must return a valid frame")
            self.assertEqual(res.shape, img.shape)
        finally:
            self.worker._compute_stage2_umat = original_compute_umat


if __name__ == '__main__':
    unittest.main()
