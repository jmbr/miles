# __all__ = ['TimestepperMOIL']
#
# from miles import BaseTimestepper
#
#
# class TimestepperMOIL(BaseTimestepper):
#     """Runs a trajectory from an initial phase space point until it hits a
#     milestone.
#
#     """
#     def __init__(self, reporter, configuration,
#                  max_chunks=default.max_chunks):
#         super().__init__(reporter, configuration, max_chunks)
#
#         self.chunks_class = ChunksMOIL
#
#     def find_transitions(self, chunk):
#         """Find transitions in sequences of DCD/DVD files.
#
#         """
#         out = chunk.output_name
#         dcd_file_name = path_ext_join(out, 'dcd')
#         dvd_file_name = path_ext_join(out, 'dvd')
#
#         transitions = []
#
#         with DCDReader(dcd_file_name) as dcd, DVDReader(dvd_file_name) as dvd:
#             for x, v in zip(dcd, dvd):
#                 p = PhaseSpacePoint(x, v, self._projection_mapping)
#
#                 self._current_time += self.configuration.time_step_length
#
#                 transition = self._reporter.next(p.colvars,
#                                                  self._current_time)
#
#                 if transition:
#                     self.save(transition, dcd, dvd)
#                     self._current_time = 0
#                     transitions.append(transition)
#
#         return transitions
#
#     def save(self, transition, dcd, dvd):
#         """Store files associated to a transition into the database."""
#         file_name = get_random_name()
#         transition.file_name = file_name
#
#         database_dir = self.configuration.database_dir
#         dcd_file_name = path_ext_join(database_dir, file_name, 'dcd')
#         dvd_file_name = path_ext_join(database_dir, file_name, 'dvd')
#
#         dcd.save_current_frame_to(dcd_file_name)
#         dvd.save_current_frame_to(dvd_file_name)
