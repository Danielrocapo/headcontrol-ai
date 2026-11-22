import unittest

from v2.engine import HeadController, MouthScrollToggle, PoseSample, WinkDetector, relative_pose


class HeadControlTests(unittest.TestCase):
    def test_pose_is_normalized(self):
        result = relative_pose((0.5, 0.6), (0.4, 0.4), (0.6, 0.4))
        self.assertAlmostEqual(result.x, 0.0)
        self.assertAlmostEqual(result.y, 1.0)

    def test_calibration_requires_30_samples(self):
        controller = HeadController()
        sample = PoseSample(0.0, 0.2)
        for _ in range(29):
            self.assertFalse(controller.observe(sample))
        self.assertFalse(controller.observe(sample))
        self.assertIsNotNone(controller.neutral)
        self.assertTrue(controller.observe(sample))

    def test_cursor_stays_still_in_center(self):
        controller = HeadController()
        for _ in range(30):
            controller.observe(PoseSample(0.0, 0.5))
        for _ in range(20):
            self.assertEqual(controller.move(PoseSample(0.0, 0.5), 0.033), (0, 0, 0))

    def test_right_head_turn_produces_horizontal_movement(self):
        controller = HeadController()
        for _ in range(30):
            controller.observe(PoseSample(0.0, 0.5))
        x = 0
        for _ in range(20):
            dx, dy, scroll = controller.move(PoseSample(-0.28, 0.5), 0.033)
            x += dx
            self.assertEqual(dy, 0)
            self.assertEqual(scroll, 0)
        self.assertGreater(x, 0)

    def test_wink_does_not_trigger_on_both_eyes(self):
        wink = WinkDetector()
        self.assertIsNone(wink.update(0.95, 0.95, 0.0))
        self.assertIsNone(wink.update(0.0, 0.0, 0.3))

    def test_wink_is_click_on_release(self):
        wink = WinkDetector()
        self.assertIsNone(wink.update(0.9, 0.1, 1.0))
        self.assertEqual(wink.update(0.0, 0.0, 1.3), "left")
        self.assertIsNone(wink.update(0.9, 0.1, 1.5))
        self.assertIsNone(wink.update(0.0, 0.0, 1.8))

    def test_open_mouth_toggles_scroll_after_one_second(self):
        mouth = MouthScrollToggle()
        self.assertFalse(mouth.update(0.8, 0.0))
        self.assertTrue(mouth.update(0.8, 1.1))
        self.assertTrue(mouth.scroll)
        self.assertFalse(mouth.update(0.8, 1.2))
        mouth.update(0.0, 1.3)
        mouth.update(0.8, 2.5)
        self.assertTrue(mouth.update(0.8, 3.6))
        self.assertFalse(mouth.scroll)


if __name__ == "__main__":
    unittest.main()
