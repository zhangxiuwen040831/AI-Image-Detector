import unittest
import shutil
import tempfile
import csv
import json
from pathlib import Path
from PIL import Image
from scripts.data_fusion import DataMerger, DataAnalyzer

class TestDataFusion(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.cifake_dir = self.test_dir / "cifake"
        self.artifact_dir = self.test_dir / "artifact"
        self.target_dir = self.test_dir / "target"
        
        self.cifake_dir.mkdir()
        self.artifact_dir.mkdir()
        self.target_dir.mkdir()
        
        # Setup CIFAKE mock
        (self.cifake_dir / "train" / "REAL").mkdir(parents=True)
        (self.cifake_dir / "train" / "FAKE").mkdir(parents=True)
        self._create_dummy_image(self.cifake_dir / "train" / "REAL" / "img1.jpg")
        self._create_dummy_image(self.cifake_dir / "train" / "FAKE" / "img2.jpg")
        
        # Setup Artifact mock
        (self.artifact_dir / "coco").mkdir()
        self._create_dummy_image(self.artifact_dir / "coco" / "img3.jpg")
        with open(self.artifact_dir / "coco" / "metadata.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "image_path", "target", "category"])
            writer.writerow(["img3.jpg", "img3.jpg", "0", "coco"]) # Real
            
        (self.artifact_dir / "big_gan").mkdir()
        self._create_dummy_image(self.artifact_dir / "big_gan" / "img4.jpg")
        with open(self.artifact_dir / "big_gan" / "metadata.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "image_path", "target", "category"])
            writer.writerow(["img4.jpg", "img4.jpg", "6", "big_gan"]) # Fake

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_image(self, path):
        img = Image.new("RGB", (32, 32), color="red")
        img.save(path)

    def test_analysis(self):
        analyzer = DataAnalyzer([self.cifake_dir, self.artifact_dir])
        stats = analyzer.analyze()
        self.assertIn("cifake", stats)
        self.assertIn("coco", stats)
        self.assertEqual(stats["cifake"]["count"], 2)
        self.assertEqual(stats["coco"]["count"], 1)

    def test_merge(self):
        merger = DataMerger([self.cifake_dir, self.artifact_dir], self.target_dir)
        stats = merger.merge()
        
        # Assert files exist
        self.assertTrue((self.target_dir / "train" / "real" / "img1.jpg").exists())
        self.assertTrue((self.target_dir / "train" / "fake" / "img2.jpg").exists())
        
        # Assert Artifact files exist (train or test depending on random split, but with 1 file it goes to test usually if ratio < 1.0, wait ratio is 0.8)
        # Actually split logic: i < split_idx. If len=1, split_idx = int(1*0.8) = 0. So i=0 is not < 0. So it goes to test.
        # But let's check both
        artifact_real = list((self.target_dir / "train" / "real").glob("img3.jpg")) + list((self.target_dir / "test" / "real").glob("img3.jpg"))
        self.assertEqual(len(artifact_real), 1)
        
        artifact_fake = list((self.target_dir / "train" / "fake").glob("img4.jpg")) + list((self.target_dir / "test" / "fake").glob("img4.jpg"))
        self.assertEqual(len(artifact_fake), 1)
        
        # Check manifest
        with open(self.target_dir / "fusion_manifest.json", "r") as f:
            manifest = json.load(f)
            self.assertEqual(len(manifest["files"]), 4) # 2 cifake + 2 artifact

    def test_rollback(self):
        merger = DataMerger([self.cifake_dir], self.target_dir)
        merger.merge()
        self.assertTrue((self.target_dir / "train" / "real" / "img1.jpg").exists())
        
        merger.rollback()
        self.assertFalse((self.target_dir / "train" / "real" / "img1.jpg").exists())
        self.assertFalse((self.target_dir / "fusion_manifest.json").exists())

if __name__ == "__main__":
    unittest.main()
