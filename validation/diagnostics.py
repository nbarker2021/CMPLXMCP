"""
System Diagnostics
==================
Real-time health checks and troubleshooting tools.
"""

import os
import sys
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import platform


@dataclass
class HealthReport:
    """System health report."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    overall_status: str = "unknown"  # "healthy", "degraded", "critical"
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "components": self.components,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
        }


class SystemDiagnostics:
    """
    Comprehensive system diagnostics.
    
    Usage:
        diag = SystemDiagnostics()
        report = diag.run_full_diagnostics()
        
        if report.overall_status != "healthy":
            for rec in report.recommendations:
                print(f"Recommendation: {rec}")
    """
    
    def __init__(self):
        self.checks = []
    
    def run_full_diagnostics(self) -> HealthReport:
        """Run complete system diagnostics."""
        report = HealthReport()
        
        # System Resources
        report.components["system_resources"] = self._check_system_resources()
        
        # Python Environment
        report.components["python_env"] = self._check_python_environment()
        
        # Dependencies
        report.components["dependencies"] = self._check_dependencies()
        
        # File System
        report.components["filesystem"] = self._check_filesystem()
        
        # MCP Server
        report.components["mcp_server"] = self._check_mcp_server()
        
        # Universal System
        report.components["universal_system"] = self._check_universal_system()
        
        # AGRM+MDHG
        report.components["agrm_mdhg"] = self._check_agrm_mdhg()
        
        # Calculate overall status
        report.overall_status = self._calculate_overall_status(report.components)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            status = "healthy"
            issues = []
            
            if cpu_percent > 80:
                status = "warning"
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 85:
                status = "warning"
                issues.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                status = "critical"
                issues.append(f"Low disk space: {disk.percent}% used")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "issues": issues,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment."""
        try:
            return {
                "status": "healthy",
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        required = {
            "numpy": "NumPy",
            "mcp": "MCP Protocol",
        }
        
        optional = {
            "psutil": "System monitoring",
            "matplotlib": "Visualization",
        }
        
        results = {
            "required": {},
            "optional": {},
            "status": "healthy",
            "missing": [],
        }
        
        for module, name in required.items():
            try:
                __import__(module)
                results["required"][module] = {"status": "ok", "name": name}
            except ImportError:
                results["required"][module] = {"status": "missing", "name": name}
                results["missing"].append(module)
                results["status"] = "critical"
        
        for module, name in optional.items():
            try:
                __import__(module)
                results["optional"][module] = {"status": "ok", "name": name}
            except ImportError:
                results["optional"][module] = {"status": "missing", "name": name}
        
        return results
    
    def _check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem structure."""
        required_paths = [
            "mcp_os",
            "mcp_os/server",
            "mcp_os/client",
            "mcp_os/universal",
            "mcp_os/agrm_mdhg_integration",
        ]
        
        results = {
            "status": "healthy",
            "paths": {},
            "missing": [],
        }
        
        for path in required_paths:
            exists = os.path.isdir(path)
            results["paths"][path] = exists
            if not exists:
                results["missing"].append(path)
                results["status"] = "critical"
        
        return results
    
    def _check_mcp_server(self) -> Dict[str, Any]:
        """Check MCP server status."""
        try:
            # Check if server module loads
            from ..server import create_server
            
            return {
                "status": "healthy",
                "module_loads": True,
                "tools_available": 35,  # Approximate
            }
        except Exception as e:
            return {
                "status": "error",
                "module_loads": False,
                "error": str(e),
            }
    
    def _check_universal_system(self) -> Dict[str, Any]:
        """Check Universal System components."""
        try:
            from ..universal import UniversalTranslator
            
            return {
                "status": "healthy",
                "translator_loads": True,
                "components": ["translator", "crystal", "temporal", "identity"],
            }
        except Exception as e:
            return {
                "status": "error",
                "translator_loads": False,
                "error": str(e),
            }
    
    def _check_agrm_mdhg(self) -> Dict[str, Any]:
        """Check AGRM+MDHG components."""
        try:
            from ..agrm_mdhg_integration import Planet, PlanetNetwork
            
            return {
                "status": "healthy",
                "modules_load": True,
                "components": ["MDHG", "CA", "AGRM", "Planet", "Network"],
            }
        except Exception as e:
            return {
                "status": "error",
                "modules_load": False,
                "error": str(e),
            }
    
    def _calculate_overall_status(self, components: Dict[str, Any]) -> str:
        """Calculate overall system status."""
        statuses = [c.get("status", "unknown") for c in components.values()]
        
        if "critical" in statuses:
            return "critical"
        elif "error" in statuses:
            return "degraded"
        elif "warning" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    def _generate_recommendations(self, report: HealthReport) -> List[str]:
        """Generate recommendations based on health report."""
        recommendations = []
        
        # Resource recommendations
        resources = report.components.get("system_resources", {})
        if resources.get("status") == "warning":
            for issue in resources.get("issues", []):
                recommendations.append(f"Resource: {issue}")
        
        # Dependency recommendations
        deps = report.components.get("dependencies", {})
        for missing in deps.get("missing", []):
            recommendations.append(f"Install missing dependency: {missing}")
        
        # Filesystem recommendations
        fs = report.components.get("filesystem", {})
        for missing in fs.get("missing", []):
            recommendations.append(f"Create missing directory: {missing}")
        
        # Component-specific recommendations
        for name, component in report.components.items():
            if component.get("status") == "error":
                error = component.get("error", "Unknown error")
                recommendations.append(f"Fix {name}: {error}")
        
        return recommendations
    
    def quick_check(self) -> str:
        """Quick health check - returns status string."""
        report = self.run_full_diagnostics()
        return report.overall_status
