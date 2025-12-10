"""Unit tests for serialization and data preparation functions in mosaic_planner."""

import tempfile
import shutil
from pathlib import Path
from io import BytesIO
import zipfile
import csv

from mosaic_planner.planner import (
    serialize_plan_with_data,
    deserialize_plan_with_data,
    prepare_file_data_for_transmission,
)
from mosaic_config.state import Data, FileDefinition, DataType, Model, ModelType, Plan


class TestSerializeDeserializePlanWithData:
    """Test serialization and deserialization of Plan and Data objects."""
    
    def test_serialize_deserialize_round_trip_compressed(self):
        """Test that serialize/deserialize works correctly with compression."""
        # Create test plan and data
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        
        # Create test file definition with binary data
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('test.csv', 'col1,col2\nval1,val2\n')
        zip_data = zip_buffer.getvalue()
        
        file_def = FileDefinition(
            location="test_file.csv",
            data_type=DataType.CSV,
            is_segmentable=True,
            binary_data=zip_data,
        )
        data = Data(file_definitions=[file_def])
        
        # Serialize
        serialized = serialize_plan_with_data(plan, data, compress=True)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize
        deserialized_plan, deserialized_data = deserialize_plan_with_data(serialized, compressed=True)
        
        # Verify plan
        assert deserialized_plan.id == plan.id
        # Model objects are not persisted - only model_id is saved
        assert deserialized_plan.model_id == plan.model_id
        assert deserialized_plan.model_id == model.id
        assert len(deserialized_plan.stats_data) == len(plan.stats_data)
        assert len(deserialized_plan.distribution_plan) == len(plan.distribution_plan)
        
        # Verify data
        assert len(deserialized_data.file_definitions) == 1
        assert deserialized_data.file_definitions[0].location == file_def.location
        assert deserialized_data.file_definitions[0].data_type == file_def.data_type
        assert deserialized_data.file_definitions[0].is_segmentable == file_def.is_segmentable
        assert deserialized_data.file_definitions[0].binary_data == file_def.binary_data
    
    def test_serialize_deserialize_round_trip_uncompressed(self):
        """Test that serialize/deserialize works correctly without compression."""
        # Create test plan and data
        model = Model(name="test_model", model_type=ModelType.TRANSFORMER)
        plan = Plan(
            stats_data=[],
            distribution_plan=[],
            model=model,
        )
        
        file_def = FileDefinition(
            location="test_file.txt",
            data_type=DataType.TEXT,
            is_segmentable=False,
        )
        data = Data(file_definitions=[file_def])
        
        # Serialize without compression
        serialized = serialize_plan_with_data(plan, data, compress=False)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize without compression
        deserialized_plan, deserialized_data = deserialize_plan_with_data(serialized, compressed=False)
        
        # Verify plan
        assert deserialized_plan.id == plan.id
        # Model objects are not persisted - only model_id is saved
        assert deserialized_plan.model_id == plan.model_id
        assert deserialized_plan.model_id == model.id
        
        # Verify data
        assert len(deserialized_data.file_definitions) == 1
        assert deserialized_data.file_definitions[0].location == file_def.location
    
    def test_serialize_deserialize_multiple_file_definitions(self):
        """Test serialization with multiple file definitions."""
        model = Model(name="test_model")
        plan = Plan(
            stats_data=[],
            distribution_plan=[],
            model=model,
        )
        
        # Create multiple file definitions
        file_defs = []
        for i in range(3):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f'test_{i}.csv', f'col1,col2\nval{i},val{i+1}\n')
            zip_data = zip_buffer.getvalue()
            
            file_def = FileDefinition(
                location=f"test_file_{i}.csv",
                data_type=DataType.CSV,
                is_segmentable=True,
                binary_data=zip_data,
            )
            file_defs.append(file_def)
        
        data = Data(file_definitions=file_defs)
        
        # Serialize and deserialize
        serialized = serialize_plan_with_data(plan, data, compress=True)
        deserialized_plan, deserialized_data = deserialize_plan_with_data(serialized, compressed=True)
        
        # Verify all file definitions
        assert len(deserialized_data.file_definitions) == 3
        for i, file_def in enumerate(deserialized_data.file_definitions):
            assert file_def.location == f"test_file_{i}.csv"
            assert file_def.binary_data == file_defs[i].binary_data


class TestPrepareFileDataForTransmission:
    """Test prepare_file_data_for_transmission function."""
    
    def test_prepare_csv_file_segmentable(self):
        """Test preparing CSV file data for transmission (segmentable)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_file = temp_path / "test.csv"
            
            # Create test CSV file
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['col1', 'col2'])
                for i in range(10):
                    writer.writerow([f'val1_{i}', f'val2_{i}'])
            
            file_def = FileDefinition(
                location="test.csv",
                data_type=DataType.CSV,
                is_segmentable=True,
            )
            
            segment_info = {
                "start_row": 2,
                "end_row": 7,
            }
            
            result = prepare_file_data_for_transmission(file_def, segment_info, str(temp_path))
            
            # Verify result is zipped bytes
            assert isinstance(result, bytes)
            assert len(result) > 0
            
            # Verify it's a valid zip file
            zip_buffer = BytesIO(result)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                assert len(zip_file.namelist()) > 0
    
    def test_prepare_csv_file_non_segmentable(self):
        """Test preparing CSV file data for transmission (non-segmentable)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_file = temp_path / "test.csv"
            
            # Create test CSV file
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['col1', 'col2'])
                for i in range(5):
                    writer.writerow([f'val1_{i}', f'val2_{i}'])
            
            file_def = FileDefinition(
                location="test.csv",
                data_type=DataType.CSV,
                is_segmentable=False,
            )
            
            segment_info = {}  # Not used for non-segmentable
            
            result = prepare_file_data_for_transmission(file_def, segment_info, str(temp_path))
            
            # Verify result is zipped bytes
            assert isinstance(result, bytes)
            assert len(result) > 0
            
            # Verify it's a valid zip file containing the full file
            zip_buffer = BytesIO(result)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                names = zip_file.namelist()
                assert len(names) == 1
                assert names[0] == "test.csv"
                
                # Verify content
                content = zip_file.read(names[0]).decode('utf-8')
                assert 'col1,col2' in content
                assert 'val1_0' in content
    
    def test_prepare_directory_non_segmentable(self):
        """Test preparing directory data for transmission (non-segmentable)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_dir = temp_path / "test_dir"
            test_dir.mkdir()
            
            # Create files in directory
            (test_dir / "file1.txt").write_text("content1")
            (test_dir / "file2.txt").write_text("content2")
            subdir = test_dir / "subdir"
            subdir.mkdir()
            (subdir / "file3.txt").write_text("content3")
            
            file_def = FileDefinition(
                location="test_dir",
                data_type=DataType.DIR,
                is_segmentable=False,
            )
            
            segment_info = {}  # Not used for non-segmentable
            
            result = prepare_file_data_for_transmission(file_def, segment_info, str(temp_path))
            
            # Verify result is zipped bytes
            assert isinstance(result, bytes)
            assert len(result) > 0
            
            # Verify it's a valid zip file containing all files
            zip_buffer = BytesIO(result)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                names = zip_file.namelist()
                assert len(names) == 3
                assert "file1.txt" in names
                assert "file2.txt" in names
                assert "subdir/file3.txt" in names
                
                # Verify content
                assert zip_file.read("file1.txt").decode('utf-8') == "content1"
                assert zip_file.read("file2.txt").decode('utf-8') == "content2"
                assert zip_file.read("subdir/file3.txt").decode('utf-8') == "content3"
    
    def test_prepare_image_segment(self):
        """Test preparing image segment for transmission."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for i in range(5):
                (images_dir / f"image_{i}.jpg").write_bytes(b"fake_image_data_" + str(i).encode())
            
            file_def = FileDefinition(
                location="images",
                data_type=DataType.IMAGE,
                is_segmentable=True,
            )
            
            segment_info = {
                "image_indices": [0, 2, 4],
            }
            
            result = prepare_file_data_for_transmission(file_def, segment_info, str(temp_path))
            
            # Verify result is zipped bytes
            assert isinstance(result, bytes)
            assert len(result) > 0
            
            # Verify it's a valid zip file
            zip_buffer = BytesIO(result)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                assert len(zip_file.namelist()) > 0
    
    def test_prepare_text_segment(self):
        """Test preparing text segment for transmission."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            text_file = temp_path / "test.txt"
            text_file.write_text("This is a test file with some content for segmentation.")
            
            file_def = FileDefinition(
                location="test.txt",
                data_type=DataType.TEXT,
                is_segmentable=True,
            )
            
            segment_info = {
                "start_char": 10,
                "end_char": 30,
            }
            
            result = prepare_file_data_for_transmission(file_def, segment_info, str(temp_path))
            
            # Verify result is zipped bytes
            assert isinstance(result, bytes)
            assert len(result) > 0
            
            # Verify it's a valid zip file
            zip_buffer = BytesIO(result)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                names = zip_file.namelist()
                assert len(names) > 0
                # Verify segment content
                content = zip_file.read(names[0]).decode('utf-8')
                assert len(content) == 20  # end_char - start_char
    
    def test_prepare_nonexistent_file_raises_error(self):
        """Test that preparing data for non-existent file raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            file_def = FileDefinition(
                location="nonexistent.csv",
                data_type=DataType.CSV,
                is_segmentable=True,
            )
            
            segment_info = {"start_row": 0, "end_row": 10}
            
            try:
                prepare_file_data_for_transmission(file_def, segment_info, str(temp_path))
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "does not exist" in str(e) or "Path does not exist" in str(e)

