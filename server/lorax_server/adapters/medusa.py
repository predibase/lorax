# @dataclass
# class MedusaConfig(AdapterConfig):
#     medusa_num_heads: int
#     medusa_num_layers: int

#     def map_weights_for_model(
#         self, adapter_weights: Dict, weight_names: Tuple[str],
#     ) -> Tuple[ModuleMap, Set[str]]:
#         return adapter_weights, set(weight_names)

#     @classmethod
#     def load(cls, config: dict) -> "MedusaConfig":
#         return cls(
#             base_model_name_or_path=config["base_model_name_or_path"],
#             medusa_num_heads=config["medusa_num_heads"],
#             medusa_num_layers=config["medusa_num_layers"],
#         )
