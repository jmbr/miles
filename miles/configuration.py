"""Configuration file management.

"""


__all__ = ['Configuration', 'ConfigurationError']

import logging
import os
import tempfile


class ConfigurationError(Exception):
    """Error parsing configuration file."""
    pass


class Configuration:
    """Manage configuration file settings.

    """
    def __init__(self, simulation_dir=None, temp_dir=None,
                 md_template=None, md_template_reset_velocities=None,
                 shell=None, command=None, database_file=None,
                 reactant_distribution=None, product_milestone=None,
                 steps_per_chunk=None, time_step_length=None,
                 logging_level=None, colvars_file=None, memoize=None):
        self.simulation_dir = simulation_dir
        self.temp_dir = temp_dir
        self.md_template = md_template
        self.md_template_reset_velocities = md_template_reset_velocities
        self.shell = shell
        self.command = command
        self.database_file = database_file
        self.reactant_distribution = reactant_distribution
        self.product_milestone = product_milestone
        self.steps_per_chunk = steps_per_chunk
        self.time_step_length = time_step_length
        self.logging_level = logging_level
        self.colvars_file = colvars_file
        self.memoize = memoize

    def __repr__(self):
        return ('{}(simulation_dir={!r}, '
                'temp_dir={!r}, '
                'md_template={!r}, '
                'md_template_reset_velocities={!r}, '
                'shell={!r}, '
                'command={!r}, '
                'database_file={!r}, '
                'reactant_distribution={!r}, '
                'product_milestone={!r}, '
                'steps_per_chunk={!r}, '
                'time_step_length={!r}, '
                'logging_level={!r}, '
                'colvars_file={!r}, '
                'memoize={!r}'
                .format(self.__class__.__name__,
                        self.simulation_dir,
                        self.temp_dir,
                        self.md_template,
                        self.md_template_reset_velocities,
                        self.shell,
                        self.command,
                        self.database_file,
                        self.reactant_distribution,
                        self.product_milestone,
                        self.steps_per_chunk,
                        self.time_step_length,
                        self.logging_level,
                        self.colvars_file,
                        self.memoize))

    def parse(self, file_name):
        """Parse configuration file.

        Raises
        ------
        ConfigurationError
            In case some configuration key is wrong.
        """
        import configparser

        extint = configparser.ExtendedInterpolation()
        cp = configparser.ConfigParser(interpolation=extint)

        # We prefix a default section name so that the user does not
        # have to.
        config_file_contents = '[DEFAULT]\n'
        with open(file_name, 'r') as cf:
            config_file_contents += cf.read()

        cp.read_string(config_file_contents)

        s = cp['DEFAULT']

        self.simulation_dir = expanduser(s.get('simulation_dir'))
        self.temp_dir = expanduser(s.get('temp_dir',
                                         tempfile.gettempdir()))

        self.md_template = expanduser(s.get('md_template'))
        self.md_template_reset_velocities \
            = expanduser(s.get('md_template_reset_velocities'))

        self.shell = expanduser(s.get('shell', '/bin/sh'))
        self.command = s.get('command', raw=True)

        self.database_file = expanduser(s.get('database_file'))
        self.reactant_distribution = expanduser(s.get('reactant_distribution'))
        pair = s.get('product_milestone')
        if pair is not None:
            self.product_milestone = tuple(int(x.strip())
                                           for x in pair.split(','))
        self.steps_per_chunk = s.getint('steps_per_chunk')
        self.time_step_length = s.getfloat('time_step_length')

        self.logging_level = s.getint('logging_level', logging.INFO)

        self.colvars_file = s.get('colvars_file')

        memoize = s.get('memoize', 'true').lower()
        if memoize in {'true', 'on'}:
            self.memoize = True
        elif memoize in {'false', 'off'}:
            self.memoize = False
        else:
            raise ConfigurationError('Invalid value for memoize: {}'
                                     .format(memoize))

    def __str__(self) -> str:
        items = sorted(self.__dict__.items(), key=lambda x: x[0])
        lines = ['Configuration options in use:']
        for key, value in items:
            lines.append('  {}: {!r}'.format(key, value))
        return '\n'.join(lines)


def expanduser(file_name):
    """Expand user directory in path.

    """
    if file_name is not None:
        return os.path.expanduser(file_name)
