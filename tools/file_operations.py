"""
File Operations Tool for Agentix agents.

This tool provides file system operations including:
- File reading and writing
- Directory operations
- File search and filtering
- Security validation
"""

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator

from .base import BaseTool, ToolConfig, ToolResult, ToolStatus


class FileOperationsConfig(ToolConfig):
    """Configuration for file operations tool."""
    
    # Security settings
    allowed_directories: List[str] = Field(default_factory=list)
    blocked_directories: List[str] = Field(default_factory=lambda: ["/etc", "/sys", "/proc"])
    allowed_extensions: List[str] = Field(default_factory=lambda: [".txt", ".json", ".csv", ".md"])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Operation settings
    create_directories: bool = True
    overwrite_files: bool = False
    backup_on_overwrite: bool = True
    
    @validator('max_file_size')
    def validate_max_file_size(cls, v):
        if v < 1024 or v > 100 * 1024 * 1024:  # 1KB to 100MB
            raise ValueError("max_file_size must be between 1KB and 100MB")
        return v


class FileOperationsTool(BaseTool):
    """
    File Operations Tool for file system interactions.
    
    This tool provides:
    - Safe file reading and writing
    - Directory operations
    - File search and filtering
    - Security validation and sandboxing
    """
    
    def __init__(self, config: FileOperationsConfig):
        super().__init__(config)
        self.file_config = config
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute file operation with the given parameters."""
        
        operation = parameters.get('operation', '')
        file_path = parameters.get('path', '')
        
        if not operation:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error="Operation is required",
                tool_name=self.name,
                tool_version=self.version
            )
        
        try:
            # Validate file path security
            if file_path and not self._validate_path_security(file_path):
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error="File path failed security validation",
                    tool_name=self.name,
                    tool_version=self.version
                )
            
            # Execute based on operation type
            if operation == "read":
                result = await self._read_file(file_path, parameters)
            elif operation == "write":
                result = await self._write_file(file_path, parameters)
            elif operation == "list":
                result = await self._list_directory(file_path, parameters)
            elif operation == "create_directory":
                result = await self._create_directory(file_path, parameters)
            elif operation == "delete":
                result = await self._delete_file(file_path, parameters)
            elif operation == "copy":
                result = await self._copy_file(parameters)
            elif operation == "move":
                result = await self._move_file(parameters)
            elif operation == "search":
                result = await self._search_files(parameters)
            elif operation == "info":
                result = await self._get_file_info(file_path)
            else:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error=f"Unknown operation: {operation}",
                    tool_name=self.name,
                    tool_version=self.version
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                tool_name=self.name,
                tool_version=self.version
            )
            
        except Exception as e:
            self.logger.error(f"File operation failed: {str(e)}")
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"File operation failed: {str(e)}",
                tool_name=self.name,
                tool_version=self.version
            )
    
    async def _read_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Read file content."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.file_config.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.file_config.max_file_size})")
        
        # Read file based on parameters
        encoding = parameters.get('encoding', 'utf-8')
        max_lines = parameters.get('max_lines')
        start_line = parameters.get('start_line', 1)
        
        try:
            if max_lines:
                # Read specific lines
                with open(path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                    end_line = min(start_line + max_lines - 1, len(lines))
                    content = ''.join(lines[start_line-1:end_line])
            else:
                # Read entire file
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
            
            return {
                'content': content,
                'file_path': str(path),
                'file_size': file_size,
                'encoding': encoding,
                'lines_read': len(content.splitlines()) if content else 0
            }
            
        except UnicodeDecodeError:
            # Try reading as binary
            with open(path, 'rb') as f:
                binary_content = f.read()
            
            return {
                'content': binary_content.hex(),
                'file_path': str(path),
                'file_size': file_size,
                'encoding': 'binary',
                'is_binary': True
            }
    
    async def _write_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to file."""
        path = Path(file_path)
        content = parameters.get('content', '')
        encoding = parameters.get('encoding', 'utf-8')
        append = parameters.get('append', False)
        
        # Create parent directories if needed
        if self.file_config.create_directories:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and handle overwrite
        if path.exists() and not self.file_config.overwrite_files and not append:
            if self.file_config.backup_on_overwrite:
                backup_path = path.with_suffix(path.suffix + '.backup')
                shutil.copy2(path, backup_path)
            else:
                raise FileExistsError(f"File already exists: {file_path}")
        
        # Write content
        mode = 'a' if append else 'w'
        with open(path, mode, encoding=encoding) as f:
            f.write(content)
        
        file_size = path.stat().st_size
        
        return {
            'file_path': str(path),
            'bytes_written': len(content.encode(encoding)),
            'file_size': file_size,
            'encoding': encoding,
            'append_mode': append
        }
    
    async def _list_directory(self, dir_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents."""
        path = Path(dir_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")
        
        # List parameters
        recursive = parameters.get('recursive', False)
        include_hidden = parameters.get('include_hidden', False)
        file_pattern = parameters.get('pattern', '*')
        
        files = []
        directories = []
        
        if recursive:
            items = path.rglob(file_pattern)
        else:
            items = path.glob(file_pattern)
        
        for item in items:
            if not include_hidden and item.name.startswith('.'):
                continue
            
            item_info = {
                'name': item.name,
                'path': str(item),
                'size': item.stat().st_size if item.is_file() else 0,
                'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                'is_file': item.is_file(),
                'is_directory': item.is_dir()
            }
            
            if item.is_file():
                files.append(item_info)
            else:
                directories.append(item_info)
        
        return {
            'directory': str(path),
            'files': files,
            'directories': directories,
            'total_files': len(files),
            'total_directories': len(directories)
        }
    
    async def _create_directory(self, dir_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create directory."""
        path = Path(dir_path)
        parents = parameters.get('parents', True)
        exist_ok = parameters.get('exist_ok', True)
        
        path.mkdir(parents=parents, exist_ok=exist_ok)
        
        return {
            'directory': str(path),
            'created': True,
            'parents_created': parents
        }
    
    async def _delete_file(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file or directory."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {file_path}")
        
        recursive = parameters.get('recursive', False)
        
        if path.is_file():
            path.unlink()
            return {
                'path': str(path),
                'deleted': True,
                'type': 'file'
            }
        elif path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()  # Only works if directory is empty
            
            return {
                'path': str(path),
                'deleted': True,
                'type': 'directory',
                'recursive': recursive
            }
    
    async def _copy_file(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Copy file or directory."""
        source = parameters.get('source', '')
        destination = parameters.get('destination', '')
        
        if not source or not destination:
            raise ValueError("Both source and destination are required")
        
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        if source_path.is_file():
            shutil.copy2(source_path, dest_path)
        else:
            shutil.copytree(source_path, dest_path)
        
        return {
            'source': str(source_path),
            'destination': str(dest_path),
            'copied': True,
            'type': 'file' if source_path.is_file() else 'directory'
        }
    
    async def _move_file(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Move file or directory."""
        source = parameters.get('source', '')
        destination = parameters.get('destination', '')
        
        if not source or not destination:
            raise ValueError("Both source and destination are required")
        
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        shutil.move(source_path, dest_path)
        
        return {
            'source': str(source_path),
            'destination': str(dest_path),
            'moved': True
        }
    
    async def _search_files(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for files."""
        search_path = parameters.get('path', '.')
        pattern = parameters.get('pattern', '*')
        content_search = parameters.get('content_search', '')
        max_results = parameters.get('max_results', 100)
        
        path = Path(search_path)
        results = []
        
        for file_path in path.rglob(pattern):
            if file_path.is_file() and len(results) < max_results:
                file_info = {
                    'path': str(file_path),
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                # Content search if specified
                if content_search:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content_search.lower() in content.lower():
                                file_info['content_match'] = True
                                results.append(file_info)
                    except (UnicodeDecodeError, PermissionError):
                        continue
                else:
                    results.append(file_info)
        
        return {
            'search_path': str(path),
            'pattern': pattern,
            'content_search': content_search,
            'results': results,
            'total_found': len(results)
        }
    
    async def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {file_path}")
        
        stat = path.stat()
        
        return {
            'path': str(path),
            'name': path.name,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'is_file': path.is_file(),
            'is_directory': path.is_dir(),
            'is_symlink': path.is_symlink(),
            'permissions': oct(stat.st_mode)[-3:],
            'extension': path.suffix
        }
    
    def _validate_path_security(self, file_path: str) -> bool:
        """Validate file path for security."""
        try:
            path = Path(file_path).resolve()
            path_str = str(path)
            
            # Check blocked directories
            for blocked_dir in self.file_config.blocked_directories:
                if path_str.startswith(blocked_dir):
                    self.logger.warning(f"Access to blocked directory: {blocked_dir}")
                    return False
            
            # Check allowed directories (if specified)
            if self.file_config.allowed_directories:
                allowed = False
                for allowed_dir in self.file_config.allowed_directories:
                    if path_str.startswith(allowed_dir):
                        allowed = True
                        break
                
                if not allowed:
                    self.logger.warning(f"Access outside allowed directories: {path_str}")
                    return False
            
            # Check file extension (if file)
            if path.suffix and self.file_config.allowed_extensions:
                if path.suffix.lower() not in self.file_config.allowed_extensions:
                    self.logger.warning(f"File extension not allowed: {path.suffix}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Path validation error: {str(e)}")
            return False
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate file operation parameters."""
        operation = parameters.get('operation')
        if not operation:
            return False
        
        valid_operations = [
            'read', 'write', 'list', 'create_directory', 
            'delete', 'copy', 'move', 'search', 'info'
        ]
        
        if operation not in valid_operations:
            return False
        
        # Operation-specific validation
        if operation in ['read', 'write', 'delete', 'info', 'list', 'create_directory']:
            if 'path' not in parameters:
                return False
        
        if operation in ['copy', 'move']:
            if 'source' not in parameters or 'destination' not in parameters:
                return False
        
        if operation == 'write':
            if 'content' not in parameters:
                return False
        
        return True
