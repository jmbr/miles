"""Check database consistency.

"""

import logging
import zipfile

from miles import Database, Simulation


def fsck(simulation: Simulation) -> int:
    """Verify database integrity.

    Parameters
    ----------
    simulation : Simulation
         Object containing all the simulation parameters.

    Returns
    -------
    valid : bool
         True if the database passes the consistency checks, false
         otherwise.

    """
    database = simulation.database
    conn = database.conn
    zip_file = database.zip_file
    zip_file_name = database.zip_file_name

    try:
        zip_file.testzip()
    except zipfile.BadZipFile as exc:
        logging.error('Invalid zip file {!r}: {}'
                      .format(database.zip_file_name, exc))
        return False

    valid = True

    c1 = conn.cursor()
    rows_transitions = c1.execute('SELECT transition_id FROM transitions')
    for row_transitions in rows_transitions:
        tid = row_transitions[0]
        c2 = conn.cursor()
        rows_files = c2.execute('SELECT file_name FROM files '
                                'WHERE transition_id = ?', row_transitions)
        row_files = rows_files.fetchall()
        if not row_files:
            logging.error('Transition with id {} does not have any associated'
                          ' files.'.format(tid))
            valid = False
        else:
            for row in row_files:
                file_name = row[0]
                try:
                    zip_file.getinfo(file_name)
                except KeyError:
                    logging.error('File {!r} corresponding to transition {}'
                                  ' not found'.format(file_name, tid))
                    valid = False

    if valid:
        logging.info('Everything is OK.')

    return valid
