from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

otel_config = {
    "exporter_otlp_endpoint": "http://localhost:4317",
    "exporter_otlp_insecure": True,
}

otlp_trace_exporter = OTLPSpanExporter(
    endpoint=otel_config["exporter_otlp_endpoint"],
    insecure=otel_config["exporter_otlp_insecure"],
)

# Observability Initialization
provider = TracerProvider()
processor = BatchSpanProcessor(otlp_trace_exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("xcube.observability")


def set_attributes(*attrs):
    span = trace.get_current_span()
    combined = {}
    for attr in attrs:
        combined.update(attr)
    span.set_attributes(combined)
