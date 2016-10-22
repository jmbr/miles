"""Database of first hitting points.

This is a registry of first hitting points. The Distribution class
relies on this database to keep track of the files containing phase
space points, etc.

"""

__all__ = ['DatabaseError', 'DatabaseAlreadyExistsError', 'DatabaseFileNotFoundError', 'Database', 'load_database', 'save_database', 'make_database']  # noqa: E501

import logging
import os
import shutil
import sqlite3
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from miles import (Anchor, CollectiveVariables, Distribution, Distributions, Milestone, Milestones, Transition, get_random_name)  # noqa: E501


ANCHORS_SQL = """\
DROP TABLE IF EXISTS anchors;
CREATE TABLE anchors (
anchor_id INTEGER PRIMARY KEY,
{}
);
"""

COLLECTIVE_VARIABLES_SQL = 'cv{:d} DOUBLE NOT NULL'

TRANSITIONS_SQL = """\
DROP TABLE IF EXISTS transitions;
CREATE TABLE transitions (
transition_id INTEGER PRIMARY KEY,
final_anchor0 INTEGER NOT NULL,
final_anchor1 INTEGER NOT NULL,
initial_anchor0 INTEGER,
initial_anchor1 INTEGER,
lag_time DOUBLE,
{}
);

CREATE INDEX transition_index ON transitions (transition_id);
"""

FILES_SQL = """\
DROP TABLE IF EXISTS files;
CREATE TABLE files (
file_id INTEGER PRIMARY KEY,
transition_id INTEGER NOT NULL,
file_name VARCHAR(128) NOT NULL
);

CREATE INDEX file_index ON files (transition_id);
"""


class DatabaseError(Exception):
    """Database-related exception.

    """
    pass


class DatabaseFileNotFoundError(DatabaseError):
    """File not found in database.

    """
    pass


class DatabaseAlreadyExistsError(DatabaseError):
    """Failure due to insertion of duplicate transition.

    """
    pass


class Database:
    """Database of transitions.

    This class acts as a database of transitions, storing their
    metadata and the first hitting points in phase space.  The
    metadata (initial and final milestones, lag times, etc.) is stored
    in a sqlite3 data store while the actual phase space points are
    kept in a hierarchy of subdirectories.

    Attributes
    ----------
    file_name
        Name for the database file.
    collective_variables
        Space of collective variables.
    zip_file_name
        Name of the zip file containing the phase space points.
    zip_file
        File handle of the zip file.
    conn
        Connection to the sqlite3 database.
    dim
        Dimension of the space of collective variables.
    anchors
        List of anchors determining the tessellation into milestones.
    milestones
        Factory of milestones.

    """
    def __init__(self, file_name: str,
                 collective_variables: CollectiveVariables) -> None:
        self.file_name, self.zip_file_name = get_file_names(file_name)

        self.collective_variables = collective_variables

        self.zip_file = None    # type: Optional[zipfile.ZipFile]
        self.conn = None        # type: Optional[sqlite3.Connection]

        self.dim = None         # type: Optional[int]

        self.anchors = []       # type: List[Anchor]
        self.milestones = Milestones()

        self._open()
        self._load_anchors()

    def __repr__(self) -> str:
        return '{}({!r})'.format(self.__class__.__name__, self.file_name)

    def __del__(self) -> None:
        self._close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['conn'] = None
        state['zip_file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def _open(self) -> None:
        """Open database.

        """
        # Open associated files.
        assert self.conn is None
        self.conn = sqlite3.connect(self.file_name, 60)
        self.conn.set_trace_callback(logging.debug)

        assert self.zip_file is None
        self.zip_file = zipfile.ZipFile(self.zip_file_name, mode='a')

    def _load_anchors(self) -> None:
        # Load anchors and set up milestone factory.
        c = self.conn.cursor()
        rows = c.execute('SELECT * FROM anchors')
        for row in rows.fetchall():
            anchor_id = row[0]-1
            coordinates = np.array(row[1:])
            anchor = Anchor(self.collective_variables, coordinates, anchor_id)
            self.anchors.append(anchor)
            self.milestones.append_anchor(anchor)
            if not self.dim:
                self.dim = coordinates.shape[0]

    def _close(self) -> None:
        """Close database and related files.

        """
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

        if self.zip_file is not None:
            self.zip_file.close()
            self.zip_file = None

        self.anchors = []
        self.milestones = Milestones()

    def reload_zip_file(self, mode='a') -> None:
        """Reload ZIP file.

        Parameters
        ----------
        mode : {'a', 'r'}
            Mode for opening the ZIP file: 'a' means writing/appending
            while 'r' means reading.

        Notes
        -----
        This must be done by the MPI services before starting from
        transitions gathered in the previous milestone.

        """
        if self.zip_file is not None:
            self.zip_file.close()

        self.zip_file = zipfile.ZipFile(self.zip_file_name, mode=mode)

    def reload_zip_file_for_reading(self) -> None:
        """Reload ZIP file for reading."""
        self.reload_zip_file(mode='r')

    def save(self) -> None:
        """Write database to disk.

        """
        self.conn.commit()

    def insert_anchor(self, anchor: Anchor) -> None:
        """Insert a new anchor.

        Parameters
        ----------
        anchor : Anchor
            Anchor to insert into the database.

        Raises
        ------
        DatabaseError
            In case there is an inconsistency about the anchors.

        """
        def make_sql_statement(coordinates):
            colvarstr = ', '.join(map(str, coordinates))
            return ('INSERT INTO anchors VALUES (NULL, {})'
                    .format(colvarstr))

        dim = anchor.coordinates.shape[0]

        if not self.dim:
            self.dim = dim
        elif self.dim != dim:
            raise DatabaseError('A {}-dimensional anchor does not belong to a '
                                '{}-dimensional space of collective variables'
                                .format(dim, self.dim))

        cursor = self.conn.cursor()
        cursor.execute(make_sql_statement(anchor.coordinates))
        self.conn.commit()

        self.milestones.append_anchor(anchor)

    def insert_anchors_from_csv(self, csv_file_name: str) -> None:
        """Insert anchors defined in a CSV file.

        The format for each line of the CSV file is as follows:

          anchor-index, collective-variable-0, collective-variable-1,
          ...

        The field anchor-index is an integer starting at zero.


        Parameters
        ----------
        csv_file_name : str
            File name of the text file containing the transitions in
            comma-separated values (CSV) format.

        Raises
        ------
        DatabaseError
            If there are problems (such as inconsistencies in the
            dimensions, etc.) inserting the anchors.

        """
        import csv

        with open(csv_file_name, 'r') as f:
            for row in csv.reader(f):
                i = int(row[0])
                c = np.array([float(val) for val in row[1:]])

                anchor = Anchor(self.collective_variables, c, i)

                self.insert_anchor(anchor)

    def _commit(self) -> None:
        """Commit a transaction to the database.

        """
        try:
            self.conn.commit()
        except sqlite3.DatabaseError:
            logging.warning('There was a problem committing to the database')
            # Be careful: we roll the database back but the directory
            # hierarchy is not rolled back.
            self.conn.rollback()

    def insert_transition(self, transition: Transition,
                          delete_files: bool = True) -> None:
        """Insert transition into the database.

        Adds a new transition to the corresponding table in the
        database.

        Parameters
        ----------
        transition : Transition
            A transition to store in the database. Upon storage, the
            transition's transition_id field will be updated with the
            assigned transition identifier in the database. The
            transition is expected to have a list of auxiliary file
            names associated to it.
        delete_files : bool
            Whether to erase or not the auxiliary files associated to
            the transition after they have been copied to the
            directory hierarchy.

        Raises
        ------
        DatabaseError
            If there is a problem with the insertion (for example, if
            the files associated with the transition do not exist,
            etc.)

        """
        self._insert_transition(transition, delete_files, commit=True)

    def _insert_file_name(self, transition_id: int, file_name: str,
                          delete_files: bool) -> None:
        """Insert auxiliary file into database.

        File names are assumed to be unique throughout the
        database.

        """
        c = self.conn.cursor()

        base_name = os.path.basename(file_name)

        self.zip_file.write(file_name, arcname=base_name)
        if delete_files:
            os.unlink(file_name)

        # Insert the new file into the files table.
        c.execute('INSERT INTO files (transition_id, file_name) '
                  'VALUES (?, ?)', (transition_id, base_name,))

    def _insert_transition(self, transition: Transition,
                           delete_files: bool, commit: bool) -> None:
        """Insert transition into database.

        """
        def get_anchor_indices() -> Tuple[List[int], List[int]]:
            # Obtain anchors for final milestone.
            final = transition.final_milestone
            a = sorted([a.index + 1 for a in final.anchors])

            # Obtain anchors for initial milestone, if known.
            initial = transition.initial_milestone
            if initial is not None:
                b = sorted([b.index + 1 for b in initial.anchors])
            else:
                b = [None, None]

            return a, b

        def get_coordinates() -> str:
            # Coordinates on the space of collective variables.
            coordinates = ', '.join(map(str, transition.colvars))
            assert self.dim == len(transition.colvars)
            return coordinates

        # We begin by doing some sanity checks.
        if transition.transition_id is not None:
            msg = '"{}" is already in the database'.format(transition)
            raise DatabaseAlreadyExistsError(msg)

        if not transition.file_names:
            raise DatabaseError('Transition without auxiliary files.')

        # Insert the new transition into the database. We must format
        # the query appropriately so that it uses all the collective
        # variables.
        cursor = self.conn.cursor()

        a, b = get_anchor_indices()
        coordinates = get_coordinates()
        cursor.execute('INSERT INTO transitions VALUES '
                       '(NULL, ?, ?, ?, ?, ?, {})'.format(coordinates),
                       (a[0], a[1], b[0], b[1], transition.lag_time))

        transition.transition_id = cursor.lastrowid

        # Insert auxiliary files into the database.
        for file_name in transition.file_names:
            self._insert_file_name(transition.transition_id,
                                   file_name, delete_files)
        if delete_files:
            transition.remove_files()

        if commit:
            self._commit()

    def insert_transitions(self, transitions: List[Transition],
                           delete_files: bool = True) -> None:
        for transition in transitions:
            self._insert_transition(transition, delete_files, commit=False)
        self._commit()

    def insert_transitions_from_csv(self, csv_file_name: str) -> None:
        """Insert transitions into the database from CSV file.

        The format for each line of the CSV file is as follows:

          final-anchor-0, final-anchor-1, initial-anchor-0,
          initial-anchor-1, lag-time, file_name, collective-variable-0,
          collective-variable-1, ...

        Where the fields beginning with final-anchor- and
        initial-anchor- contain the two anchors needed to uniquely
        identify a milestone.

        The initial-anchor- fields as well as the lag-time field may
        be left empty.

        """
        import csv
        import math

        ms = self.milestones

        final = slice(0, 2)
        initial = slice(2, 4)

        with open(csv_file_name, 'r') as f:
            for row in csv.reader(f):
                ff = tuple(map(int, row[final]))
                final_milestone = ms.make_from_indices(ff[0], ff[1])

                initial_milestone = None
                try:
                    ii = tuple(map(int, row[initial]))
                    initial_milestone = ms.make_from_indices(ii[0], ii[1])
                except ValueError:
                    pass

                # Lag time can be empty or NaN.
                lag_time = None
                if row[4].strip() != '':
                    lag_time = float(row[4].split()[0])
                    if math.isnan(lag_time):
                        lag_time = None

                file_name = row[5]
                colvars = np.array(row[6:], dtype=np.float64)

                transition = Transition(final_milestone, initial_milestone,
                                        colvars, lag_time, file_name)
                self._insert_transition(transition, delete_files=False,
                                        commit=False)

        self._commit()

        self.milestones = ms

    def delete_transition(self, transition: Transition) -> None:
        """Delete a transition from the database.

        """
        tid = transition.transition_id
        if not tid:
            raise DatabaseError('Attempted to delete an unknown transition.')

        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM transitions WHERE transition_id = ?', (tid,))  # noqa: E501
        cursor.execute('DELETE FROM files WHERE transition_id = ?', (tid,))
        # self.conn.commit()

    def extract_files(self, transition: Transition, dir_name: str) -> None:
        """Extract files associated to a phase space point.

        This extracts all the auxiliary files associated to the given
        transition inside the dir_name directory.

        Parameters
        ----------
        transition : Transition
            Transition whose auxiliary files are to be extracted.
        dir_name : str
            Name of the directory where to extract `file_name` into.

        Raises
        ------
        DatabaseFileNotFoundError
            In case there is a problem extracting any of the files
            associated to the transition.

        Notes
        -----
        The transition object will be altered to reflect the location
        of the files.

        """
        transition_id = transition.transition_id
        if transition_id is None:
            raise DatabaseError('Invalid transition identifier for "{}"'
                                .format(transition))

        cursor = self.conn.cursor()

        results = cursor.execute('SELECT file_name FROM files '
                                 'WHERE transition_id = ?', (transition_id,))

        random_name = get_random_name(dir_name)

        output_file_names = []
        for row in results.fetchall():
            # We construct the name of the output file attempting to
            # give it the extension of the original file name for
            # clarity.
            file_name = os.path.basename(row[0])

            parts = file_name.split(os.extsep)
            ext = parts[1] if len(parts) == 2 else None

            output_file_name = random_name
            if ext:
                output_file_name = os.extsep.join([output_file_name, ext])
            if os.path.exists(output_file_name):
                raise DatabaseError('Unable to create {!r}. File already '
                                    'exists.'.format(output_file_name))

            # We extract the file from the ZIP archive to a temporary
            # directory. This is done to avoid problems with two
            # processes extracting a file from the archive and then
            # opening or erasing it at the same time.
            temp_dir_name = get_random_name(dir_name)
            try:
                extracted_file_name = self.zip_file.extract(file_name,
                                                            path=temp_dir_name)
            except KeyError:
                msg = ('Cannot find file {!r} associated to "{}"'
                       .format(file_name, transition))
                raise DatabaseFileNotFoundError(msg)

            # Finally, we place the file at its definitive location
            # and we remove the temporary directory.
            shutil.move(extracted_file_name, output_file_name)
            os.removedirs(temp_dir_name)

            output_file_names.append(output_file_name)

        transition.set_files(output_file_names)

    def to_distributions(self, reset_velocities: bool) -> Distributions:
        """Return the content of the database as a Distributions object.

        This method allows the user to construct the initial
        distributions from a database.

        Parameters
        ----------
        reset_velocities
            True, if we reset the velocities in the
            transitions. False, otherwise.

        Returns
        -------
        dists
            An instance of the Distributions class containing the
            entries in the database.

        """
        dd = {}              # type: Dict[Milestone, List[Transition]]
        ms = self.milestones

        c = self.conn.cursor()
        rows = c.execute('SELECT * FROM transitions')
        for row in rows.fetchall():
            transition_id = row[0]

            final_milestone = ms.make_from_indices(row[1]-1, row[2]-1)
            if row[3] is None or row[4] is None:
                initial_milestone = None
            else:
                initial_milestone = ms.make_from_indices(row[3]-1, row[4]-1)
            lag_time = row[5]
            coordinates = np.array(row[6:])

            transition = Transition(final_milestone, initial_milestone,
                                    coordinates, lag_time,
                                    reset_velocities=reset_velocities,
                                    transition_id=transition_id)

            transitions = dd.setdefault(final_milestone, list())
            transitions.append(transition)

        dists = Distributions()
        for milestone in dd.keys():
            dists[milestone] = Distribution(dd[milestone])

        return dists

    def get_entries(self) -> Dict[Milestone, List[Transition]]:
        """Return all valid transitions as a dictionary.

        Returns
        -------
        d : Dict[Milestone, List[Transition]]
            Dictionary of lists of transitions indexed by final
            milestone.  Transitions without a declared initial
            milestone are not included.

        """
        ms = self.milestones

        d = {}               # type: Dict[Milestone, List[Transition]]

        c = self.conn.cursor()
        rows = c.execute('SELECT * FROM transitions '
                         'WHERE initial_anchor0 IS NOT NULL '
                         'AND initial_anchor1 IS NOT NULL')

        for row in rows.fetchall():
            transition_id = row[0]
            final = ms.make_from_indices(row[1]-1, row[2]-1)
            initial = ms.make_from_indices(row[3]-1, row[4]-1)
            lag_time = row[5]
            colvars = np.array(row[6:])
            transition = Transition(final, initial, colvars,
                                    lag_time, reset_velocities=False,
                                    transition_id=transition_id)

            entries = d.setdefault(final, [])
            entries.append(transition)

        return d


def load_database(file_name: str,
                  collective_variables: CollectiveVariables) -> Database:
    """Load a database from the file system.

    Parameters
    ----------
    file_name : str
        Name of file containing the database.
    collective_variables : CollectiveVariables
        Space of collective variables.

    Returns
    -------
    database : Database
        An instance of the Database class initialized with the
        information stored in `file_name`.

    Raises
    ------
    OSError

    """
    logging.debug('Loading database {!r}...'.format(file_name))
    database = Database(file_name, collective_variables)
    logging.debug('Done.')

    return database


def save_database(database: Database) -> None:
    """Save a database to the file system.

    Parameters
    ----------
    database
        An instance of the Database class containing the data.

    Raises
    ------
    RuntimeError

    """
    logging.debug('Loading database {!r}...'.format(database.file_name))
    database.save()
    logging.debug('Done.')


def make_database(file_name: str,
                  collective_variables: CollectiveVariables) -> Database:
    """Create a new database.

    This method creates (or resets) a database, setting up all the
    required tables in the process.

    Parameters
    ----------
    file_name : str
        Name of the database. Two files with the name followed by the
        suffixes 'db' and 'zip' will be created in the file system.
    collective_variables : CollectiveVariables
        The space of collective variables.

    """
    db_file_name, zip_file_name = get_file_names(file_name)

    # Backup previous files if they exist.
    extsep = os.path.extsep

    try:
        db_file_name_bak = extsep.join([db_file_name, 'bak'])
        shutil.move(db_file_name, db_file_name_bak)
    except FileNotFoundError:
        pass

    try:
        zip_file_name_bak = extsep.join([zip_file_name, 'bak'])
        shutil.move(zip_file_name, zip_file_name_bak)
    except FileNotFoundError:
        pass

    def make_sql_statement(dim):
        cvt = COLLECTIVE_VARIABLES_SQL
        cvs = ',\n'.join(cvt.format(i) for i in range(dim))
        return '\n'.join([ANCHORS_SQL.format(cvs),
                          TRANSITIONS_SQL.format(cvs),
                          FILES_SQL])

    conn = sqlite3.connect(db_file_name)
    with conn:
        cursor = conn.cursor()
        dim = collective_variables.dimension
        cursor.executescript(make_sql_statement(dim))

    conn.close()

    return Database(file_name, collective_variables)


def get_file_names(file_name: str) -> Tuple[str, str]:
    """Return file names from prefix.

    """
    file_name_parts = file_name.split(os.path.extsep)
    if len(file_name_parts) > 1 and file_name_parts[-1] == 'db':
        file_name = file_name[0]

    db_file_name = os.path.extsep.join([file_name, 'db'])
    zip_file_name = os.path.extsep.join([file_name, 'zip'])

    return db_file_name, zip_file_name
