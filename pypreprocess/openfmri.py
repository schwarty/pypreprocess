# standard imports
import os
import glob
import warnings

# import spm preproc utilities
from .nipype_preproc_spm_utils import (do_subjects_preproc, SubjectData)
from .datasets import fetch_openfmri

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""


def preproc_dataset(data_dir, output_dir, open_output=None,
                    ignore_subjects=None, restrict_subjects=None,
                    delete_orient=False, dartel=False,
                    n_jobs=-1):
    """Main function for preprocessing a dataset with the OpenfMRI layout.

    Parameters
    ----------
    data_dir: str
        Path of input directory. If does not exist and finishes
        by a valid OpenfMRI dataset id, it will be downloaded,
        i.e., /path/to/dir/{dataset_id}.
    output_dir: str
        Path of output directory.
    open_output: str or None
        Path of the base output directory following the openfmri layout.
        If None will default to {output_dir}/../.openfmri/{dataset_id}. If
        not None will create an directory at {open_output}/{dataset_id}.
    ignore_subjects: list or None
        List of subject identifiers not to process.
    restrict_subjects: list or None
        List of subject identifiers to process.
    delete_orient: bool
        Delete orientation information in nifti files.
    dartel: bool
        Use dartel.
    n_jobs: int
        Number of parallel jobs.

    Examples
    --------
    preproc_dataset('/tmp/ds105', '/tmp/ds105_preproc ',
                    ignore_subjects=['sub002', 'sub003'],
                    delete_orient=True,
                    n_jobs=3)

    Warning
    -------
    Subjects may be excluded if some data is missing.

    Returns list of Bunch objects with fields anat, func, and subject_id
    for each preprocessed subject
    """
    parent_dir, dataset_id = os.path.split(data_dir)

    if not os.path.exists(data_dir):
        fetch_openfmri(parent_dir, dataset_id)

    ignore_subjects = [] if ignore_subjects is None else ignore_subjects

    # glob for subjects and their imaging sessions identifiers
    if restrict_subjects is None:
        subjects = [os.path.basename(x)
                    for x in glob.glob(os.path.join(data_dir, 'sub???'))]
    else:
        subjects = restrict_subjects

    subjects = sorted(subjects)

    # producer subject data
    def subject_factory():
        for subject_id in subjects:
            if subject_id in ignore_subjects:
                continue

            sessions = set()
            subject_dir = os.path.join(data_dir, subject_id)
            for session_dir in glob.glob(os.path.join(
                    subject_dir, 'BOLD', '*')):
                sessions.add(os.path.split(session_dir)[1])
            sessions = sorted(sessions)
            # construct subject data structure
            subject_data = SubjectData()
            subject_data.session_id = sessions
            subject_data.subject_id = subject_id
            subject_data.func = []

            # glob for BOLD data
            has_bad_sessions = False
            for session_id in subject_data.session_id:
                bold_dir = os.path.join(
                    data_dir, subject_id, 'BOLD', session_id)

                # glob BOLD data for this session
                func = glob.glob(os.path.join(bold_dir, "bold.nii.gz"))
                # check that this session is OK (has BOLD data, etc.)
                if not func:
                    warnings.warn(
                        'Subject %s is missing data for session %s.' % (
                        subject_id, session_id))
                    has_bad_sessions = True
                    break

                subject_data.func.append(func[0])

            # exclude subject if necessary
            if has_bad_sessions:
                warnings.warn('Excluding subject %s' % subject_id)
                continue

            # anatomical data
            subject_data.anat = os.path.join(
                data_dir, subject_id, 'anatomy', 'highres001.nii.gz')
            # pypreprocess is setup to work with non-skull stripped brain and
            # is likely to crash otherwise.
            if not os.path.exists(subject_data.anat):
                subject_data.anat = os.path.join(
                    data_dir, subject_id, 'anatomy', 'highres001_brain.nii.gz')

            # subject output_dir
            subject_data.output_dir = os.path.join(output_dir, subject_id)
            yield subject_data

    preproc = do_subjects_preproc(
        subject_factory(),
        n_jobs=n_jobs,
        dataset_id=dataset_id,
        output_dir=output_dir,
        deleteorient=delete_orient,
        dartel=dartel,
        dataset_description=DATASET_DESCRIPTION,
        preproc_params=preproc_params,
        coreg_anat_to_func=True,
        # caching=False,
        )

    _save_to_layout(data_dir, output_dir, preproc, open_output)

    return preproc


def _save_to_layout(data_dir, preproc_dir, preproc, base_output=None):
    """Function to hard link preproc data to an openfmri-like layout.
    """
    study_id = os.path.split(data_dir.rstrip('/'))[1]
    if base_output is None:
        base_output = _check_dir(
            os.path.join(os.path.split(preproc_dir)[0], '.openfmri', study_id))
    else:
        base_output = _check_dir(os.path.join(base_output, study_id))

    # index preproc with subject_id
    preproc = dict([(s.subject_id, s) for s in preproc])

    # first copy top study metadata
    models_dir = _check_dir(os.path.join(base_output, 'models', 'model001'))
    models_files = glob.glob(os.path.join(data_dir, 'models',
                                          'model001', '*.txt'))
    _link_files(models_dir, models_files)

    txt_files = glob.glob(os.path.join(data_dir, '*.txt'))
    _link_files(base_output, txt_files)

    # subject level data
    for subject_dir in glob.glob(os.path.join(preproc_dir, 'sub???')):
        subject_id = os.path.split(subject_dir)[1]

        # link onsets from data folder
        onsets = os.path.join(data_dir, subject_id, 'model', 'model001', 'onsets', '*')
        for session_dir in glob.glob(onsets):
            session_id = os.path.split(session_dir)[1]
            onsets_dir = _check_dir(os.path.join(
                base_output, subject_id,
                'model', 'model001', 'onsets', session_id))
            onsets_files = glob.glob(os.path.join(session_dir, '*.txt'))
            _link_files(onsets_dir, onsets_files)

        # link data from preproc folders
        sub_preproc = preproc[subject_id]

        anat_dir = _check_dir(os.path.join(
            base_output, subject_id,
            'model', 'model001', 'anatomy'))
        _link_file(anat_dir, sub_preproc.anat, 'highres001.nii')

        for session_id, func, motion in zip(
                sub_preproc.session_id,
                sub_preproc.func,
                sub_preproc.realignment_parameters):
            session_dir = _check_dir(os.path.join(
                base_output, subject_id,
                'model', 'model001', 'BOLD', session_id))
            _link_file(session_dir, func, 'bold.nii')
            _link_file(session_dir, motion, 'motion.txt')


def _check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _link_files(dest_dir, files):
    for f_src in files:
        f_dest = os.path.join(dest_dir, os.path.split(f_src)[1])
        if os.path.exists(f_dest):
            os.remove(f_dest)
        os.link(f_src, f_dest)


def _link_file(dest_dir, src, fname=None):
    fname = os.path.split(src)[1] if fname is None else fname
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        os.remove(dest)
    os.link(src, dest)
