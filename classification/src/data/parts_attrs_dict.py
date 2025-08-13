parts_attrs_dict = {
    "cpu": {
        "part_name": {
            "columns": {
                "name": {}
            } 
            
        },
        "part_type": "CPU"
    },
    "motherboard": {
        "part_name": {
            "columns": {
                "name": {},
                # "form_factor": {},
                # "socket": {}
            }
        },
        "part_type": "MOTHERBOARD"
    },
    "case-fan": {
        "part_name": {
            "columns": {
                "name": {},
                # "size": {
                #     "suffix": "mm",
                # },
                # "rpm": {
                #     "join": "-",
                #     "suffix": "RPM",
                # },
                # "noise_level": {
                #     "join": "-",
                #     "suffix": "dB"
                # },
                # "color": {},
                # "pwm": {
                #     "bool_format": ("PWM", "DC"),
                # }
            }
        },
        "part_type": "CASE_FAN"
    },
    "memory": {
        "part_name": {
            "columns": {
                "name": {},
                # "modules": {},
                # "speed": {
                #     "suffix": "MHz",
                # },
                # "cas_latency": {
                #     "suffix": "CL",
                # },
                # "color": {},
            }
        },
        "part_type": "RAM"
    },
    "case": {
        "part_name": {
            "columns": {
                "name": {},
                # "type": {},
                # "color": {},
                # "side_panel": {},
            }
        },
        "part_type": "CASE"  
    },
    "power-supply": {
        "part_name": {
            "columns": {
                "name": {},
                # "wattage": {
                #     "suffix": "W",
                # },
                # "efficiency": {
                #     "upper": True,
                # },
                # "modular": {
                #     "bool_format": ("Modular", "Non-Modular"),
                # },
                # "type": {},
            }
        },
        "part_type": "PSU"
    },
    "cpu-cooler": {
        "part_name": {
            "columns": {
                "name": {},
                # "size": {
                #     "suffix": "mm",
                # },
                # "rpm": {
                #     "join": "-",
                #     "suffix": "RPM",
                # },
                # "noise_level": {
                #     "join": "-",
                #     "suffix": "dB"
                # },
                # "color": {},
            }
        },
        "part_type": "CPU_COOLER"
    },
    
    "video-card": {
        "part_name": {
            "columns": {
                "name": {},
                "chipset": {},
                # "color": {}
            } 
        },
        "part_type": "GPU"
    },
    
    "internal-hard-drive": {
        "part_name": {
            "columns": {
                "name": {},
                # "capacity": {
                #     "suffix": "GB",
                # },
                # "type": {},
                # "interface": {},
                # "cache": {
                #     "bool_format": ("Cache", "No Cache"),
                # }
                
            }
        },
        "part_type": "STORAGE"
    }
}