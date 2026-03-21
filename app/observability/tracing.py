import os
import time
from functools import wraps
from typing import Callable, Any
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode


# Initialize tracer provider
resource = Resource.create({"service.name": "rag-agent"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Configure exporters
# Console exporter for development
console_exporter = ConsoleSpanExporter()
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(console_exporter)
)

# Optional: OTLP exporter for production (e.g., Jaeger, Honeycomb)
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        headers={"x-api-key": os.getenv("OTEL_API_KEY", "")}
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )


def trace_function(span_name: str = None):
    """
    Decorator to trace function execution.
    
    Args:
        span_name: Optional custom span name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = span_name or func.__name__
            
            with tracer.start_as_current_span(name) as span:
                start_time = time.time()
                
                try:
                    # Add function parameters as attributes
                    span.set_attribute("function.name", func.__name__)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record success
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                    
                finally:
                    # Record duration
                    duration = time.time() - start_time
                    span.set_attribute("duration_ms", duration * 1000)
        
        return wrapper
    return decorator


class TracingContext:
    """Context manager for manual span creation."""
    
    def __init__(self, span_name: str, **attributes):
        self.span_name = span_name
        self.attributes = attributes
        self.span = None
        self.start_time = None
    
    def __enter__(self):
        self.span = tracer.start_span(self.span_name)
        self.start_time = time.time()
        
        # Set initial attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            # Record duration
            duration = time.time() - self.start_time
            self.span.set_attribute("duration_ms", duration * 1000)
            
            # Record error if any
            if exc_type is not None:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.end()


def trace_request(span_name: str = "http_request"):
    """
    Decorator specifically for tracing HTTP requests.
    Tracks request duration and status.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                start_time = time.time()
                
                try:
                    span.set_attribute("http.method", "POST")
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    span.set_attribute("http.status_code", 200)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    span.set_attribute("http.status_code", 500)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("request.duration_ms", duration * 1000)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                start_time = time.time()
                
                try:
                    span.set_attribute("http.method", "POST")
                    
                    result = func(*args, **kwargs)
                    
                    span.set_attribute("http.status_code", 200)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    span.set_attribute("http.status_code", 500)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("request.duration_ms", duration * 1000)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_llm_call(model: str, prompt_tokens: int = None, completion_tokens: int = None):
    """
    Trace LLM API calls with specific attributes.
    
    Args:
        model: The model being called
        prompt_tokens: Number of prompt tokens (optional)
        completion_tokens: Number of completion tokens (optional)
    """
    span = trace.get_current_span()
    
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.provider", "openai")
    
    if prompt_tokens:
        span.set_attribute("llm.prompt_tokens", prompt_tokens)
    if completion_tokens:
        span.set_attribute("llm.completion_tokens", completion_tokens)
    if prompt_tokens and completion_tokens:
        span.set_attribute("llm.total_tokens", prompt_tokens + completion_tokens)


def trace_retrieval(query: str, top_k: int, results_count: int):
    """
    Trace retrieval operations.
    
    Args:
        query: The search query
        top_k: Number of results requested
        results_count: Actual number of results returned
    """
    span = trace.get_current_span()
    
    span.set_attribute("retrieval.query_length", len(query))
    span.set_attribute("retrieval.top_k", top_k)
    span.set_attribute("retrieval.results_count", results_count)
    span.set_attribute("retrieval.method", "faiss")