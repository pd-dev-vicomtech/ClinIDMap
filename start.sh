#!/usr/bin/env bash
 uvicorn application.web.main:create_app --host 0.0.0.0 --port 5858 