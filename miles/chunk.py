__all__ = ['Chunk']

import os
import string
import subprocess
from tempfile import NamedTemporaryFile

from miles import Configuration


class Chunk:
    """Chunk of a short trajectory.

    We refer to a trajectory started at a certain milestone and
    stopped at a neighboring milestone as a short trajectory. Short
    trajectories are comprised of chunks, which are pieces of
    trajectories obtained by running the external MD engine (and
    stored in DCD and DVD files).

    Attributes
    ----------
    input_name : str
        Input name to use in the MD engine. XXX Give examples.
    output_name : str
        Output name to use in the MD engine. XXX Give examples.
    template : str
        File containing the template file for the MD program.
    configuration : Configuration
        General configuration settings for the simulation.
    random_seed : int
        Seed for the random number generator used by the MD engine.
    content : str
        Evaluated template passed to the MD code.
    command_line : List[str]
        Evaluated template containing the command line to invoke the
        MD code.
    status : int
        Exit status of the MD code.
    stdin, stdout, stderr
        Standard input, output, and error for the MD code.

    """
    def __init__(self, configuration: Configuration, input_name: str,
                 output_name: str, template: str, random_seed: int) -> None:
        self.input_name = input_name
        self.output_name = output_name

        self.simulation_dir = configuration.simulation_dir
        self.template = os.path.join(self.simulation_dir, template)

        self.configuration = configuration

        self.random_seed = random_seed

        self.content, self.command_line, self.status = None, None, None
        self.stdin, self.stdout, self.stderr = None, None, None

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, {!r}, {!r}, {!r})'
                .format(self.__class__.__name__,
                        self.configuration,
                        self.input_name,
                        self.output_name,
                        self.template,
                        self.random_seed))

    def run(self) -> int:
        """Run command to obtain chunk of timesteps.

        Returns
        -------

        status : int
            Exit status of the MD command.

        """
        def interpolate_md_template(keywords):
            with open(self.template, 'r') as tf:
                tmp = string.Template(tf.read())
                return bytes(tmp.safe_substitute(keywords), 'utf-8')

        def interpolate_command_line_template(keywords):
            tmp = string.Template(self.configuration.command)
            command_line = str(tmp.safe_substitute(keywords))
            return [self.configuration.shell, '-c', command_line]

        # We must use simulation_dir due to NAMD2.
        with NamedTemporaryFile(dir=self.simulation_dir) as f:
            keywords = {**self.__dict__,
                        **self.configuration.__dict__,
                        'simulation_file': f.name}  # PEP 448

            self.content = interpolate_md_template(keywords)
            f.write(self.content)
            f.flush()

            self.command_line = interpolate_command_line_template(keywords)

            proc = subprocess.Popen(self.command_line,
                                    executable=self.configuration.shell,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

            self.stdout, self.stderr = proc.communicate(self.stdin)

            self.status = proc.returncode

        return self.status
