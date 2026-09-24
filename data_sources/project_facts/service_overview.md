# Public service overview

This Agent can connect to an external image-generation service over HTTP. The
service is kept separate from this repository; this repository contains the
conversation, retrieval, session, and integration layers only.

## User-facing capabilities

- Accept a text description and an optional line-art image.
- Submit a generation task and poll its status.
- Show returned candidate images and the service-recommended result.
- Query recent generation history.
- Report service availability and queue status.

## Interface boundary

The service endpoint is configured with \`PUPPET_API_BASE\`. The Agent uses these
product-level endpoints:

- \`POST /api/generate\` to create a task
- \`GET /api/tasks/{task_id}\` to poll a task
- \`GET /api/history\` to read recent tasks
- \`GET /api/health\` to check availability

The public Agent does not include model implementation, model weights, training
data, private preprocessing, service-side scoring details, or research
evaluation artifacts. Those remain in the separately managed service.

## Input handling

Supported line-art uploads are PNG, JPEG, WEBP, and BMP files up to 5 MB. The
Agent validates the local file before submitting it and never stores user
uploads in Git.
