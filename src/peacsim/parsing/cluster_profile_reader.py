from peacsim.parsing import parsing_utils as parsing_utils


class ClusterProfileReader:
    def __init__(self, file_source):
        self.cluster_profile_config = parsing_utils.init_config_parser(file_source)
        self._node_ids = self.cluster_profile_config.sections()
        self._node_count = len(self._node_ids)
        self._all_idle_watts = {node_id: self.cluster_profile_config.getfloat(node_id, 'idle_watts')
                                for node_id
                                in self._node_ids}

    @property
    def node_count(self):
        return self._node_count

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def all_idle_watts(self):
        return self._all_idle_watts

    def to_dict(self):
        return {
            "node_ids": self._node_ids,
            "node_count": self._node_count,
            "all_idle_watts": self._all_idle_watts
        }
