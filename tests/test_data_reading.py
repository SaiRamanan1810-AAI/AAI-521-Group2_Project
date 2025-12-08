import os
import unittest

try:
    from PIL import Image  # noqa: F401
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from src.data import prepare_plant_dataset, prepare_disease_dataset


@unittest.skipUnless(PIL_AVAILABLE and os.path.isdir('data/plants'), 'Pillow missing or data/plants directory not found')
class TestDataReading(unittest.TestCase):
    def test_prepare_plant_dataset(self):
        ds = prepare_plant_dataset('data/plants')
        self.assertIn('train', ds)
        self.assertIn('val', ds)
        self.assertIn('test', ds)
        self.assertIn('meta', ds)
        self.assertGreater(len(ds['train']), 0)


@unittest.skipUnless(PIL_AVAILABLE and os.path.isdir('data/diseases'), 'Pillow missing or data/diseases directory not found')
class TestDiseaseReading(unittest.TestCase):
    def test_prepare_disease_for_each_species(self):
        species_dirs = [d for d in os.listdir('data/diseases') if os.path.isdir(os.path.join('data/diseases', d))]
        for sp in species_dirs:
            ds = prepare_disease_dataset(sp, 'data/diseases')
            self.assertIn('train', ds)
            self.assertGreater(len(ds['train']), 0)


if __name__ == '__main__':
    unittest.main()
