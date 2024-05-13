from fastapi import FastAPI

from application.web import router

from fastapi.middleware.cors import CORSMiddleware


def create_app():
    app = FastAPI()
    app.include_router(router.api)

    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    

    # https://github.com/tiangolo/fastapi/issues/1921
    @app.get('/')
    def root():
        return {'message': 'Main Page for routing'}

    return app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host='0.0.0.0', port=8000)
