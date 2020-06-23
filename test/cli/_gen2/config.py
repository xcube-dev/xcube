XCUBE_API_CFG = {
    "input_configs": [
        {
            "store_id": "sentinelhub",
            "data_id": "S2L2A",
            "variable_names": [
                "B01",
                "B02",
                "B03"
            ]
        }
    ],
    "cube_config": {
        "crs": "WGS84",
        "bbox": [
            12.2,
            52.1,
            13.9,
            54.8
        ],
        "spatial_res": 0.05,
        "time_range": [
            "2018-01-01",
            None
        ],
        "time_period": "4D"
    },
    "output_config": {
        "store_id": "memory",
        "data_id": "CHL"
    },
    "callback": {
        "api_uri": "https://xcube-gen.test/api/v1/",
        "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1rUXhOMEkxTVRRMVJESTNPRUkyTTBWRFF6RTFOMEk0TUVJelJqTkZSRE5CUkRJMlJqUXhRZyJ9.eyJpc3MiOiJodHRwczovL2VkYy5ldS5hdXRoMC5jb20vIiwic3ViIjoiMTNlQmxEWjZhNHBRcjVvWTlnbTI2WVoxY29SWlRzM0pAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8veGN1YmUtZ2VuLmJyb2NrbWFubi1jb25zdWx0LmRlL2FwaS92MS8iLCJpYXQiOjE1OTI4MTMwMjksImV4cCI6MTU5Mjg5OTQyOSwiYXpwIjoiMTNlQmxEWjZhNHBRcjVvWTlnbTI2WVoxY29SWlRzM0oiLCJzY29wZSI6InJlYWQ6am9iIHN1Ym1pdDpqb2IiLCJndHkiOiJjbGllbnQtY3JlZGVudGlhbHMiLCJwZXJtaXNzaW9ucyI6WyJyZWFkOmpvYiIsInN1Ym1pdDpqb2IiXX0.sbqmxYeRM_yCP72lY8vj2-srYMfPqyJZNqjATEbS4wMrOxmRYEotufigFwr6MzT134kPDJNc1pv_xuLGI1huD_Wtw4xZAV9ooVHhII6asY1pICbLjq9LlFiKPQZs-j92gUAp7uzW-HyqIOca9bJBypl-Y8aiRVJAtssP88pFGqWNHFEE0Dl4TPEPKh3v-U11YUKI-svevD6QRpbqyf6cFrdGz_GECzSEo0scP3MvuWQrY9wF7Iy6-VZ1fXzFTzt3Ps72EqlJy5h09DqBa3K7e07N-6VSMswv_evxoCjFjzr5zyAM_HDg2LwekE7oYl1IIuBVPRD5npt4EZpBydJL4Q"
    }
}
