import pandas as pd
import pickle
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_carrot_logger

logger = get_carrot_logger(__name__)


class DatasetValidationError(Exception):
    pass


class DatasetValidator:
    SCHEMAS = {
        'ms_marco': {
            'required_columns': ['query', 'combine_passage', 'annotation_passage'],
            'optional_columns': ['optimal_C', 'optimal_lambda', 'optimal_iter'],
            'description': 'MS MARCO dataset format for CARROT'
        },
        'hotpot_qa': {
            'required_columns': ['question', 'answer', 'context'],
            'optional_columns': ['id', 'level', 'type'],
            'description': 'HotpotQA dataset format'
        },
        'general_qa': {
            'required_columns': ['query', 'context', 'answer'],
            'optional_columns': ['id', 'metadata'],
            'description': 'General QA dataset format'
        }
    }
    
    @classmethod
    def detect_dataset_type(cls, data: Union[pd.DataFrame, List[Dict]]) -> Optional[str]:
        if isinstance(data, list):
            if not data:
                logger.warning("Empty dataset provided")
                return None
            columns = set(data[0].keys())
        else:
            columns = set(data.columns)
        
        for dataset_type, schema in cls.SCHEMAS.items():
            required_cols = set(schema['required_columns'])
            if required_cols.issubset(columns):
                logger.info(f"Detected dataset type: {dataset_type}")
                return dataset_type
        
        logger.warning("Could not automatically detect dataset type")
        return None
    
    @classmethod
    def validate_schema(cls, data: Union[pd.DataFrame, List[Dict]], 
                       dataset_type: str = None) -> Dict[str, Any]:
        """Validate dataset schema and structure."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'dataset_type': dataset_type,
            'statistics': {}
        }
        
        # Convert to DataFrame for consistent processing
        if isinstance(data, list):
            if not data:
                validation_result['valid'] = False
                validation_result['errors'].append("Dataset is empty")
                return validation_result
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Auto-detect type if not provided
        if dataset_type is None:
            dataset_type = cls.detect_dataset_type(df)
            validation_result['dataset_type'] = dataset_type
        
        if dataset_type not in cls.SCHEMAS:
            validation_result['errors'].append(f"Unknown dataset type: {dataset_type}")
            validation_result['valid'] = False
            return validation_result
        
        schema = cls.SCHEMAS[dataset_type]
        
        # Check required columns
        missing_columns = set(schema['required_columns']) - set(df.columns)
        if missing_columns:
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            validation_result['valid'] = False
        
        # Check for empty values in required columns
        for col in schema['required_columns']:
            if col in df.columns:
                null_count = df[col].isna().sum()
                empty_count = (df[col] == '').sum() if df[col].dtype == 'object' else 0
                total_empty = null_count + empty_count
                
                if total_empty > 0:
                    validation_result['warnings'].append(
                        f"Column '{col}' has {total_empty} empty/null values ({total_empty/len(df)*100:.1f}%)"
                    )
        
        # Compute basic statistics
        validation_result['statistics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add column-specific statistics
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                text_lengths = df[col].astype(str).str.len()
                validation_result['statistics'][f'{col}_avg_length'] = text_lengths.mean()
                validation_result['statistics'][f'{col}_max_length'] = text_lengths.max()
            elif df[col].dtype in ['int64', 'float64']:  # Numeric columns
                validation_result['statistics'][f'{col}_mean'] = df[col].mean()
                validation_result['statistics'][f'{col}_std'] = df[col].std()
        
        return validation_result
    
    @classmethod
    def validate_file(cls, filepath: str, dataset_type: str = None) -> Dict[str, Any]:
        """Validate a dataset file."""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return {
                    'valid': False,
                    'errors': [f"File not found: {filepath}"],
                    'warnings': [],
                    'statistics': {}
                }
            
            # Load data based on file extension
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.pkl':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            elif file_ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_ext == '.csv':
                data = pd.read_csv(filepath)
            else:
                return {
                    'valid': False,
                    'errors': [f"Unsupported file format: {file_ext}"],
                    'warnings': [],
                    'statistics': {}
                }
            
            # Validate the loaded data
            validation_result = cls.validate_schema(data, dataset_type)
            validation_result['filepath'] = filepath
            validation_result['file_size_mb'] = os.path.getsize(filepath) / 1024 / 1024
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating file {filepath}: {e}")
            return {
                'valid': False,
                'errors': [f"Error loading file: {str(e)}"],
                'warnings': [],
                'statistics': {},
                'filepath': filepath
            }
    
    @classmethod
    def generate_validation_report(cls, validation_result: Dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("DATASET VALIDATION REPORT")
        report_lines.append("=" * 50)
        
        # Basic info
        if 'filepath' in validation_result:
            report_lines.append(f"File: {validation_result['filepath']}")
            report_lines.append(f"File Size: {validation_result.get('file_size_mb', 0):.2f} MB")
        
        report_lines.append(f"Dataset Type: {validation_result.get('dataset_type', 'Unknown')}")
        report_lines.append(f"Validation Status: {'✅ VALID' if validation_result['valid'] else '❌ INVALID'}")
        
        # Statistics
        stats = validation_result.get('statistics', {})
        if stats:
            report_lines.append(f"\n📊 DATASET STATISTICS:")
            report_lines.append(f"  Rows: {stats.get('total_rows', 0):,}")
            report_lines.append(f"  Columns: {stats.get('total_columns', 0)}")
            report_lines.append(f"  Memory Usage: {stats.get('memory_usage_mb', 0):.2f} MB")
        
        # Errors
        errors = validation_result.get('errors', [])
        if errors:
            report_lines.append(f"\n❌ ERRORS ({len(errors)}):")
            for error in errors:
                report_lines.append(f"  • {error}")
        
        # Warnings
        warnings = validation_result.get('warnings', [])
        if warnings:
            report_lines.append(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                report_lines.append(f"  • {warning}")
        
        if validation_result['valid'] and not warnings:
            report_lines.append(f"\n✨ Dataset is valid and ready to use!")
        
        report_lines.append("=" * 50)
        return "\n".join(report_lines)
    
    @classmethod
    def quick_check(cls, filepath: str) -> bool:
        """Quick validation check - returns True if dataset is valid."""
        result = cls.validate_file(filepath)
        return result['valid']


def validate_dataset(filepath: str, dataset_type: str = None, 
                    show_report: bool = True) -> Dict[str, Any]:
    """Convenience function to validate a dataset file."""
    validator = DatasetValidator()
    result = validator.validate_file(filepath, dataset_type)
    
    if show_report:
        report = validator.generate_validation_report(result)
        print(report)
    
    return result


def is_valid_dataset(filepath: str) -> bool:
    """Quick check if dataset is valid."""
    return DatasetValidator.quick_check(filepath)