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
        "access_token": "e4Q"
    }
}
