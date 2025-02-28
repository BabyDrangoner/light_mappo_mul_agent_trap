import numpy as np


class EnvBase:
    def __init__(self, max_col, max_row, delta_t, t_total):

        # map_size
        self.m_max_col = max_col
        self.m_max_row = max_row

        # time
        self.m_delta_t = delta_t
        self.m_t_total = t_total
        self.m_t = 0

        # done
        self.done = False

    def base_reset(self):
        self.m_t = 0
        self.done = False

    def base_step(self):
        self.m_t += 1
        if self.m_t == self.m_t_total:
            self.done = True

    def get_map_size(self):
        return self.m_max_col, self.m_max_row

    def set_map_size(self, max_col, max_row):
        self.m_max_col = max_col
        self.m_max_row = max_row

    def get_delta_t(self):
        return self.m_delta_t

    def set_delta_t(self, v):
        self.m_delta_t = v

    def get_t_total(self):
        return self.m_t_total

    def set_t_total(self, v):
        self.m_t_total = v

    def get_t(self):
        return self.m_t

    def set_t(self, v):
        self.m_t = v

    def get_done(self):
        return self.done

    def set_done(self, v):
        self.done = v
