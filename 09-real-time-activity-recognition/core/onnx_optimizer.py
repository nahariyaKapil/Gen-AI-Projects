import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import time
import json

logger = logging.getLogger(__name__)

class ONNXOptimizer:
    """
    ONNX optimization for real-time activity recognition
    """
    
    def __init__(self, models_dir: str = "models/onnx"):
        """
        Initialize ONNX optimizer
        
        Args:
            models_dir: Directory to store ONNX models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.onnx_models = {}
        self.onnx_sessions = {}
        
        # ONNX Runtime providers (prefer GPU if available)
        self.providers = self._get_available_providers()
        
        # Model configurations for ONNX conversion
        self.model_configs = {
            "I3D": {
                "input_shape": (1, 3, 16, 224, 224),
                "dynamic_axes": {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                "opset_version": 11
            },
            "SlowFast": {
                "input_shape": (1, 3, 32, 224, 224),
                "dynamic_axes": {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                "opset_version": 11
            },
            "ViViT": {
                "input_shape": (1, 3, 16, 224, 224),
                "dynamic_axes": {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                "opset_version": 11
            }
        }
        
        logger.info(f"ONNX optimizer initialized with providers: {self.providers}")
    
    def _get_available_providers(self) -> List[str]:
        """Get available ONNX Runtime providers"""
        try:
            available_providers = ort.get_available_providers()
            
            # Prefer GPU providers
            preferred_order = [
                'CUDAExecutionProvider',
                'TensorrtExecutionProvider',
                'OpenVINOExecutionProvider',
                'CPUExecutionProvider'
            ]
            
            providers = []
            for provider in preferred_order:
                if provider in available_providers:
                    providers.append(provider)
            
            # Add any remaining providers
            for provider in available_providers:
                if provider not in providers:
                    providers.append(provider)
            
            return providers
            
        except Exception as e:
            logger.error(f"Error getting ONNX providers: {str(e)}")
            return ['CPUExecutionProvider']
    
    def convert_to_onnx(self, model: torch.nn.Module, model_type: str, 
                       output_path: Optional[str] = None) -> str:
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model: PyTorch model to convert
            model_type: Type of model ("I3D", "SlowFast", "ViViT")
            output_path: Path to save ONNX model
            
        Returns:
            Path to saved ONNX model
        """
        try:
            if model_type not in self.model_configs:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            config = self.model_configs[model_type]
            
            # Generate output path if not provided
            if output_path is None:
                output_path = self.models_dir / f"{model_type.lower()}_model.onnx"
            
            # Create dummy input
            dummy_input = torch.randn(config["input_shape"])
            
            # Set model to evaluation mode
            model.eval()
            
            # Export to ONNX
            logger.info(f"Converting {model_type} model to ONNX...")
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=config["opset_version"],
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=config["dynamic_axes"]
            )
            
            # Verify the exported model
            self._verify_onnx_model(output_path)
            
            logger.info(f"Successfully converted {model_type} model to ONNX: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error converting {model_type} model to ONNX: {str(e)}")
            raise
    
    def _verify_onnx_model(self, model_path: str):
        """Verify ONNX model validity"""
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model verification passed: {model_path}")
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {str(e)}")
            raise
    
    def optimize_onnx_model(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Optimize ONNX model for better performance
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save optimized model
            
        Returns:
            Path to optimized model
        """
        try:
            if output_path is None:
                base_path = Path(model_path)
                output_path = base_path.parent / f"{base_path.stem}_optimized.onnx"
            
            # Load model
            model = onnx.load(model_path)
            
            # Apply optimizations
            from onnxruntime.tools import optimizer
            
            # Get graph optimization level
            opt_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create optimization configuration
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = opt_level
            sess_options.optimized_model_filepath = str(output_path)
            
            # Create session to trigger optimization
            session = ort.InferenceSession(model_path, sess_options, providers=self.providers)
            
            logger.info(f"Optimized ONNX model saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {str(e)}")
            # Return original path if optimization fails
            return model_path
    
    def load_onnx_model(self, model_path: str, model_type: str) -> ort.InferenceSession:
        """
        Load ONNX model for inference
        
        Args:
            model_path: Path to ONNX model
            model_type: Type of model
            
        Returns:
            ONNX Runtime inference session
        """
        try:
            # Check if already loaded
            if model_type in self.onnx_sessions:
                return self.onnx_sessions[model_type]
            
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable parallel execution
            sess_options.intra_op_num_threads = 0  # Use all available cores
            sess_options.inter_op_num_threads = 0  # Use all available cores
            
            # Create inference session
            session = ort.InferenceSession(model_path, sess_options, providers=self.providers)
            
            # Cache the session
            self.onnx_sessions[model_type] = session
            
            logger.info(f"Loaded ONNX model for {model_type}: {model_path}")
            logger.info(f"Using providers: {session.get_providers()}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def get_optimized_model(self, model_type: str) -> Optional[ort.InferenceSession]:
        """
        Get optimized ONNX model session
        
        Args:
            model_type: Type of model
            
        Returns:
            ONNX Runtime session or None if not available
        """
        try:
            if model_type in self.onnx_sessions:
                return self.onnx_sessions[model_type]
            
            # Check if ONNX model exists
            model_path = self.models_dir / f"{model_type.lower()}_model.onnx"
            optimized_path = self.models_dir / f"{model_type.lower()}_model_optimized.onnx"
            
            if optimized_path.exists():
                return self.load_onnx_model(str(optimized_path), model_type)
            elif model_path.exists():
                return self.load_onnx_model(str(model_path), model_type)
            else:
                logger.warning(f"No ONNX model found for {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting optimized model: {str(e)}")
            return None
    
    def run_inference(self, session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference using ONNX Runtime
        
        Args:
            session: ONNX Runtime session
            input_data: Input data as numpy array
            
        Returns:
            Inference results
        """
        try:
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            start_time = time.time()
            results = session.run(None, {input_name: input_data})
            inference_time = time.time() - start_time
            
            logger.debug(f"ONNX inference completed in {inference_time:.4f} seconds")
            
            return results[0]  # Return first output
            
        except Exception as e:
            logger.error(f"Error during ONNX inference: {str(e)}")
            raise
    
    def benchmark_model(self, model_type: str, num_iterations: int = 100) -> Dict:
        """
        Benchmark ONNX model performance
        
        Args:
            model_type: Type of model to benchmark
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Benchmark results
        """
        try:
            session = self.get_optimized_model(model_type)
            if session is None:
                return {"error": f"No ONNX model available for {model_type}"}
            
            # Get input shape
            input_shape = session.get_inputs()[0].shape
            
            # Replace dynamic dimensions with fixed values
            input_shape = [1 if dim == 'batch_size' else dim for dim in input_shape]
            
            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warm-up runs
            for _ in range(10):
                self.run_inference(session, dummy_input)
            
            # Benchmark runs
            times = []
            for _ in range(num_iterations):
                start_time = time.time()
                self.run_inference(session, dummy_input)
                times.append(time.time() - start_time)
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1.0 / avg_time
            
            results = {
                "model_type": model_type,
                "num_iterations": num_iterations,
                "average_time": avg_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "fps": fps,
                "providers": session.get_providers()
            }
            
            logger.info(f"Benchmark results for {model_type}: {fps:.2f} FPS")
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {str(e)}")
            return {"error": str(e)}
    
    def convert_and_optimize_model(self, model: torch.nn.Module, model_type: str) -> str:
        """
        Convert PyTorch model to ONNX and optimize it
        
        Args:
            model: PyTorch model
            model_type: Type of model
            
        Returns:
            Path to optimized ONNX model
        """
        try:
            # Convert to ONNX
            onnx_path = self.convert_to_onnx(model, model_type)
            
            # Optimize ONNX model
            optimized_path = self.optimize_onnx_model(onnx_path)
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Error in convert and optimize: {str(e)}")
            raise
    
    def get_model_info(self, model_type: str) -> Dict:
        """
        Get information about ONNX model
        
        Args:
            model_type: Type of model
            
        Returns:
            Model information
        """
        try:
            session = self.get_optimized_model(model_type)
            if session is None:
                return {"error": f"No ONNX model available for {model_type}"}
            
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            return {
                "model_type": model_type,
                "input_name": input_info.name,
                "input_shape": input_info.shape,
                "input_type": input_info.type,
                "output_name": output_info.name,
                "output_shape": output_info.shape,
                "output_type": output_info.type,
                "providers": session.get_providers()
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_models(self):
        """Clean up loaded models from memory"""
        try:
            for model_type in list(self.onnx_sessions.keys()):
                del self.onnx_sessions[model_type]
            
            self.onnx_sessions.clear()
            logger.info("Cleaned up ONNX models from memory")
            
        except Exception as e:
            logger.error(f"Error cleaning up models: {str(e)}")
    
    def list_available_models(self) -> List[str]:
        """List available ONNX models"""
        try:
            available_models = []
            
            for model_type in self.model_configs.keys():
                model_path = self.models_dir / f"{model_type.lower()}_model.onnx"
                optimized_path = self.models_dir / f"{model_type.lower()}_model_optimized.onnx"
                
                if model_path.exists() or optimized_path.exists():
                    available_models.append(model_type)
            
            return available_models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def quantize_model(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Quantize ONNX model for better performance
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save quantized model
            
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            if output_path is None:
                base_path = Path(model_path)
                output_path = base_path.parent / f"{base_path.stem}_quantized.onnx"
            
            # Apply dynamic quantization
            quantize_dynamic(
                model_input=model_path,
                model_output=str(output_path),
                weight_type=QuantType.QUInt8
            )
            
            logger.info(f"Quantized model saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error quantizing model: {str(e)}")
            return model_path
    
    def compare_performance(self, model_type: str) -> Dict:
        """
        Compare performance between PyTorch and ONNX models
        
        Args:
            model_type: Type of model to compare
            
        Returns:
            Performance comparison results
        """
        try:
            results = {
                "model_type": model_type,
                "pytorch_available": False,
                "onnx_available": False,
                "performance_improvement": 0.0
            }
            
            # Check ONNX availability
            onnx_session = self.get_optimized_model(model_type)
            if onnx_session is not None:
                results["onnx_available"] = True
                onnx_benchmark = self.benchmark_model(model_type)
                results["onnx_fps"] = onnx_benchmark.get("fps", 0)
                results["onnx_avg_time"] = onnx_benchmark.get("average_time", 0)
            
            # Note: PyTorch comparison would require the original model
            # This is a placeholder for the comparison logic
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing performance: {str(e)}")
            return {"error": str(e)}
    
    def get_system_info(self) -> Dict:
        """Get system information for ONNX Runtime"""
        try:
            return {
                "onnx_runtime_version": ort.__version__,
                "available_providers": ort.get_available_providers(),
                "device_count": len(ort.get_available_providers()),
                "current_providers": self.providers
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {"error": str(e)} 