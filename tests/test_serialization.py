"""Test suite for pickling and file path loading functionality."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fips.matrices import CovarianceMatrix, ForwardOperator, Matrix
from fips.problem import InverseProblem
from fips.vectors import Block, Vector
from tests.generate_data import generate_test_data


class TestBlockPickling:
    """Tests for Block pickling and file path loading."""

    def test_block_is_pickleable(self):
        """Test that a Block can be pickled and unpickled."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="block1")
        block = Block(series)

        pickled = pickle.dumps(block)
        unpickled = pickle.loads(pickled)

        assert unpickled.name == block.name
        assert np.allclose(unpickled.values, block.values)
        assert unpickled.index.tolist() == block.index.tolist()
    def test_block_to_file_and_from_file(self):
        """Test saving Block to and loading from file."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="block1")
        block = Block(series)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            block.to_file(temp_path)
            assert Path(temp_path).exists()

            # Load from file
            loaded_block = Block.from_file(temp_path)
            assert loaded_block.name == block.name
            assert np.allclose(loaded_block.values, block.values)
        finally:
            Path(temp_path).unlink()

    def test_block_to_file_invalid_extension(self):
        """Test that to_file rejects non-pickle extensions."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="block1")
        block = Block(series)

        with pytest.raises(ValueError, match="File extension must be"):
            block.to_file("/tmp/file.txt")

    def test_block_from_file_nonexistent(self):
        """Test that from_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            Block.from_file("/nonexistent/path/file.pkl")

class TestVectorPickling:
    """Tests for Vector pickling and file path loading."""

    def test_vector_is_pickleable(self):
        """Test that a Vector can be pickled and unpickled."""
        s1 = pd.Series([1, 2], index=["a", "b"], name="block1")
        s2 = pd.Series([3, 4], index=["c", "d"], name="block2")
        vector = Vector(name="vec", blocks=[s1, s2])

        pickled = pickle.dumps(vector)
        unpickled = pickle.loads(pickled)

        assert unpickled.name == vector.name
        assert set(unpickled.blocks.keys()) == set(vector.blocks.keys())
        assert np.allclose(unpickled.values, vector.values)
    def test_vector_to_file_and_from_file(self):
        """Test saving Vector to and loading from file."""
        s1 = pd.Series([1, 2], index=["a", "b"], name="block1")
        s2 = pd.Series([3, 4], index=["c", "d"], name="block2")
        vector = Vector(name="vec", blocks=[s1, s2])

        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file (test .pickle extension too)
            vector.to_file(temp_path)
            assert Path(temp_path).exists()

            # Load from file
            loaded_vector = Vector.from_file(temp_path)
            assert loaded_vector.name == vector.name
            assert np.allclose(loaded_vector.values, vector.values)
        finally:
            Path(temp_path).unlink()

class TestMatrixPickling:
    """Tests for Matrix, CovarianceMatrix, and ForwardOperator pickling."""

    def test_matrix_is_pickleable(self):
        """Test that a Matrix can be pickled and unpickled."""
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        matrix = Matrix(df)

        pickled = pickle.dumps(matrix)
        unpickled = pickle.loads(pickled)

        assert unpickled.shape == matrix.shape
        assert np.allclose(unpickled.values, matrix.values)

    def test_matrix_to_file_and_from_file(self):
        """Test saving Matrix to and loading from file."""
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        matrix = Matrix(df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            matrix.to_file(temp_path)
            assert Path(temp_path).exists()

            # Load from file
            loaded_matrix = Matrix.from_file(temp_path)
            assert loaded_matrix.shape == matrix.shape
            assert np.allclose(loaded_matrix.values, matrix.values)
        finally:
            Path(temp_path).unlink()

    def test_covariance_matrix_is_pickleable(self):
        """Test that a CovarianceMatrix can be pickled and unpickled."""
        df = pd.DataFrame([[1, 0.5], [0.5, 2]], index=["a", "b"], columns=["a", "b"])
        cov = CovarianceMatrix(df)

        pickled = pickle.dumps(cov)
        unpickled = pickle.loads(pickled)

        assert unpickled.shape == cov.shape
        assert np.allclose(unpickled.values, cov.values)

    def test_covariance_matrix_to_file_and_from_file(self):
        """Test saving CovarianceMatrix to and loading from file."""
        df = pd.DataFrame([[1, 0.5], [0.5, 2]], index=["a", "b"], columns=["a", "b"])
        cov = CovarianceMatrix(df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            cov.to_file(temp_path)
            assert Path(temp_path).exists()

            # Load from file
            loaded_cov = CovarianceMatrix.from_file(temp_path)
            assert loaded_cov.shape == cov.shape
            assert np.allclose(loaded_cov.values, cov.values)
        finally:
            Path(temp_path).unlink()

    def test_forward_operator_is_pickleable(self):
        """Test that a ForwardOperator can be pickled and unpickled."""
        df = pd.DataFrame(
            [[1, 0.5, 0.2], [0.5, 1, 0.3]],
            index=["obs1", "obs2"],
            columns=["state1", "state2", "state3"],
        )
        h = ForwardOperator(df)

        pickled = pickle.dumps(h)
        unpickled = pickle.loads(pickled)

        assert unpickled.shape == h.shape
        assert np.allclose(unpickled.values, h.values)

    def test_forward_operator_to_file_and_from_file(self):
        """Test saving ForwardOperator to and loading from file."""
        df = pd.DataFrame(
            [[1, 0.5, 0.2], [0.5, 1, 0.3]],
            index=["obs1", "obs2"],
            columns=["state1", "state2", "state3"],
        )
        h = ForwardOperator(df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            h.to_file(temp_path)
            assert Path(temp_path).exists()

            # Load from file
            loaded_h = ForwardOperator.from_file(temp_path)
            assert loaded_h.shape == h.shape
            assert np.allclose(loaded_h.values, h.values)
        finally:
            Path(temp_path).unlink()


class TestInverseProblemPickling:
    """Tests for InverseProblem pickling and file path loading."""

    def test_inverse_problem_is_pickleable(self):
        """Test that an InverseProblem can be pickled and unpickled."""
        test_data = generate_test_data(n_state=5, n_obs=8)
        problem = InverseProblem(
            prior=test_data["prior"],
            obs=test_data["obs"],
            forward_operator=test_data["forward_operator"],
            prior_error=test_data["prior_error"],
            modeldata_mismatch=test_data["modeldata_mismatch"],
        )

        pickle.dumps(problem)

    def test_inverse_problem_from_pickle_file(self):
        """Test pickling and unpickling an entire InverseProblem."""
        test_data = generate_test_data(n_state=5, n_obs=8)
        problem = InverseProblem(
            prior=test_data["prior"],
            obs=test_data["obs"],
            forward_operator=test_data["forward_operator"],
            prior_error=test_data["prior_error"],
            modeldata_mismatch=test_data["modeldata_mismatch"],
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(problem, f)
            temp_path = f.name

        try:
            # Load from pickle file
            with open(temp_path, "rb") as f:
                loaded_problem = pickle.load(f)
            
            assert loaded_problem.prior.n == problem.prior.n
            assert loaded_problem.obs.n == problem.obs.n
            assert np.allclose(loaded_problem.prior.values, problem.prior.values)
            assert np.allclose(loaded_problem.obs.values, problem.obs.values)
        finally:
            Path(temp_path).unlink()

    def test_inverse_problem_override_components(self):
        """Test creating InverseProblem with mix of objects and file paths."""
        test_data = generate_test_data(n_state=5, n_obs=8)
        
        # Create new prior error with different values
        new_error_data = pd.DataFrame(
            np.eye(len(test_data["prior"])) * 2.0,
            index=test_data["prior"].index,
            columns=test_data["prior"].index,
        )
        new_prior_error = CovarianceMatrix(new_error_data)

        # Save jacobian to a file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(test_data["forward_operator"], f)
            jacobian_path = f.name

        try:
            # Create problem with some components from objects, some from file
            loaded_problem = InverseProblem(
                prior=test_data["prior"],
                obs=test_data["obs"],
                forward_operator=jacobian_path,  # Load from file
                prior_error=new_prior_error,     # Use new object
                modeldata_mismatch=test_data["modeldata_mismatch"],
            )

            # Verify
            assert np.allclose(loaded_problem.prior_error.values, new_prior_error.values)
            assert np.allclose(loaded_problem.forward_operator.values, test_data["forward_operator"].values)
        finally:
            Path(jacobian_path).unlink()

    def test_inverse_problem_load_from_multiple_files(self):
        """Test loading different components from different pickle files."""
        test_data = generate_test_data(n_state=5, n_obs=8)

        # Save different components to different files
        temp_paths = {}
        for key in ["prior", "obs", "forward_operator", "prior_error", "modeldata_mismatch"]:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(test_data[key], f)
                temp_paths[key] = f.name

        try:
            # Load problem from separate files
            loaded_problem = InverseProblem(
                prior=temp_paths["prior"],
                obs=temp_paths["obs"],
                forward_operator=temp_paths["forward_operator"],
                prior_error=temp_paths["prior_error"],
                modeldata_mismatch=temp_paths["modeldata_mismatch"],
            )

            # Verify all components loaded correctly
            assert loaded_problem.prior.n == len(test_data["prior"])
            assert loaded_problem.obs.n == len(test_data["obs"])
            assert loaded_problem.forward_operator.shape == test_data["forward_operator"].shape
            assert np.allclose(loaded_problem.prior_error.values, test_data["prior_error"].values)
            assert np.allclose(
                loaded_problem.modeldata_mismatch.values, test_data["modeldata_mismatch"].values
            )
        finally:
            for path in temp_paths.values():
                Path(path).unlink()

    def test_inverse_problem_override_with_new_object(self):
        """Test that individual components can be passed as file paths."""
        test_data = generate_test_data(n_state=5, n_obs=8)
        
        # Save each component to a separate file
        temp_paths = {}
        for component, name in [
            (test_data["prior"], "prior"),
            (test_data["obs"], "obs"),
            (test_data["forward_operator"], "forward_operator"),
            (test_data["prior_error"], "prior_error"),
            (test_data["modeldata_mismatch"], "modeldata_mismatch"),
        ]:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(component, f)
                temp_paths[name] = f.name

        try:
            # Create a new prior error to override
            new_prior_error = CovarianceMatrix(
                pd.DataFrame(
                    np.eye(len(test_data["prior"])) * 0.5,
                    index=test_data["prior"].index,
                    columns=test_data["prior"].index,
                )
            )

            # Load from files, override prior_error with new object
            loaded_problem = InverseProblem(
                prior=temp_paths["prior"],
                obs=temp_paths["obs"],
                forward_operator=temp_paths["forward_operator"],
                prior_error=new_prior_error,  # New object, not a file
                modeldata_mismatch=temp_paths["modeldata_mismatch"],
            )

            # Verify override worked
            assert np.allclose(loaded_problem.prior_error.values, new_prior_error.values)
            # Verify files were loaded correctly
            assert np.allclose(loaded_problem.prior.values, test_data["prior"].values)
        finally:
            for path in temp_paths.values():
                Path(path).unlink()

    def test_inverse_problem_with_constant_from_file(self):
        """Test loading constant vector from pickle file."""
        test_data = generate_test_data(n_state=5, n_obs=8)
        constant = pd.Series([0.1] * 8, index=test_data["obs"].index)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(constant, f)
            constant_path = f.name

        try:
            problem = InverseProblem(
                prior=test_data["prior"],
                obs=test_data["obs"],
                forward_operator=test_data["forward_operator"],
                prior_error=test_data["prior_error"],
                modeldata_mismatch=test_data["modeldata_mismatch"],
                constant=constant_path,
            )

            assert isinstance(problem.constant, Vector)
            assert np.allclose(problem.constant.values, constant.values)
        finally:
            Path(constant_path).unlink()

    def test_inverse_problem_with_scalar_constant(self):
        """Test that scalar constants work (don't try to unpickle them)."""
        test_data = generate_test_data(n_state=5, n_obs=8)

        problem = InverseProblem(
            prior=test_data["prior"],
            obs=test_data["obs"],
            forward_operator=test_data["forward_operator"],
            prior_error=test_data["prior_error"],
            modeldata_mismatch=test_data["modeldata_mismatch"],
            constant=0.5,
        )

        assert problem.constant == 0.5


class TestToFileFromFile:
    """Tests for to_file() and from_file() methods."""

    def test_block_to_file_from_file(self):
        """Test Block to_file and from_file methods."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="block1")
        block = Block(series)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            block.to_file(temp_path)
            loaded = Block.from_file(temp_path)
            assert loaded.name == block.name
            assert np.allclose(loaded.values, block.values)
        finally:
            Path(temp_path).unlink()

    def test_vector_to_file_from_file(self):
        """Test Vector to_file and from_file methods."""
        s1 = pd.Series([1, 2], index=["a", "b"], name="block1")
        s2 = pd.Series([3, 4], index=["c", "d"], name="block2")
        vector = Vector(name="vec", blocks=[s1, s2])

        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
            temp_path = f.name

        try:
            vector.to_file(temp_path)
            loaded = Vector.from_file(temp_path)
            assert loaded.name == vector.name
            assert np.allclose(loaded.values, vector.values)
        finally:
            Path(temp_path).unlink()

    def test_matrix_to_file_from_file(self):
        """Test Matrix to_file and from_file methods."""
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        matrix = Matrix(df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            matrix.to_file(temp_path)
            loaded = Matrix.from_file(temp_path)
            assert loaded.shape == matrix.shape
            assert np.allclose(loaded.values, matrix.values)
        finally:
            Path(temp_path).unlink()

    def test_covariance_matrix_to_file_from_file(self):
        """Test CovarianceMatrix to_file and from_file methods."""
        df = pd.DataFrame([[1, 0.5], [0.5, 2]], index=["a", "b"], columns=["a", "b"])
        cov = CovarianceMatrix(df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            cov.to_file(temp_path)
            loaded = CovarianceMatrix.from_file(temp_path)
            assert loaded.shape == cov.shape
            assert np.allclose(loaded.values, cov.values)
        finally:
            Path(temp_path).unlink()

    def test_forward_operator_to_file_from_file(self):
        """Test ForwardOperator to_file and from_file methods."""
        df = pd.DataFrame(
            [[1, 0.5], [0.5, 1], [0.2, 0.3]],
            index=["obs1", "obs2", "obs3"],
            columns=["state1", "state2"],
        )
        h = ForwardOperator(df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            h.to_file(temp_path)
            loaded = ForwardOperator.from_file(temp_path)
            assert loaded.shape == h.shape
            assert np.allclose(loaded.values, h.values)
        finally:
            Path(temp_path).unlink()

    def test_inverse_problem_to_file_from_file(self):
        """Test InverseProblem to_file and from_file methods."""
        test_data = generate_test_data(n_state=5, n_obs=8)
        problem = InverseProblem(
            prior=test_data["prior"],
            obs=test_data["obs"],
            forward_operator=test_data["forward_operator"],
            prior_error=test_data["prior_error"],
            modeldata_mismatch=test_data["modeldata_mismatch"],
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            problem.to_file(temp_path)
            loaded = InverseProblem.from_file(temp_path)
            assert loaded.prior.n == problem.prior.n
            assert loaded.obs.n == problem.obs.n
            assert np.allclose(loaded.prior.values, problem.prior.values)
        finally:
            Path(temp_path).unlink()

    def test_to_file_invalid_extension_block(self):
        """Test that to_file rejects invalid extensions."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="block1")
        block = Block(series)

        with pytest.raises(ValueError, match="File extension must be"):
            block.to_file("/tmp/file.json")

    def test_from_file_invalid_extension_block(self):
        """Test that from_file rejects invalid extensions."""
        with pytest.raises(ValueError, match="File extension must be"):
            Block.from_file("/tmp/file.txt")

    def test_from_file_nonexistent_file(self):
        """Test that from_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            Vector.from_file("/nonexistent/path/file.pkl")

