# __all__ = ['ChunksMOIL']
#
# from typing import Optional
#
# from miles import (BaseChunks, Configuration, DCDReader, DVDReader, delete_files, get_random_name, path_ext_join)  # noqa: E501
#
#
# class ChunksMOIL(BaseChunks):
#     """Sequence of MOIL trajectory chunks.
#
#     """
#     def __init__(self, configuration: Configuration,
#                  initial_file: str, reset_velocities: bool,
#                  random_seed: Optional[int] = None) -> None:
#         super().__init__(configuration, initial_file,
#                          reset_velocities, random_seed)
#
#         self.suffixes = ('dcd', 'dvd')
#
#     def prepare(self):
#         assert self.prev_chunk is not None
#         out = self.prev_chunk.output_name
#         dcd_file_name = path_ext_join(out, 'dcd')
#         dvd_file_name = path_ext_join(out, 'dvd')
#
#         inp = get_random_name(self.configuration.temp_dir)
#         new_dcd_file_name = path_ext_join(inp, 'dcd')
#         new_dvd_file_name = path_ext_join(inp, 'dvd')
#
#         with DCDReader(dcd_file_name) as dcd, DVDReader(dvd_file_name) as dvd:
#             dcd.save_last_frame_to(new_dcd_file_name)
#             dvd.save_last_frame_to(new_dvd_file_name)
#
#         return inp
#
#     def cleanup(self):
#         delete_files(self.suffixes, self.prev_chunk.output_name)
