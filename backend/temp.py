class OPSM:

    """
    from subprocess import *

    def jarWrapper(*args):
        process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE)
        ret = []
        while process.poll() is None:
            line = process.stdout.readline()
            if line != '' and line.endswith('\n'):
                ret.append(line[:-1])
        stdout, stderr = process.communicate()
        ret += stdout.split('\n')
        if stderr != '':
            ret += stderr.split('\n')
        ret.remove('')
        return ret

    args = ['myJarFile.jar', 'arg1', 'arg2', 'argN'] # Any number of args to be passed to the jar file

    result = jarWrapper(*args)

    print result"""

    pass


class CCS(BinaryBiclusteringBase):
    """

    Args:
        thresh (float): A correlation threshold in range [0, 1]. Defaults to
            0.8.

    Attribtues:
        biclusters (list):

    """

    # Name of the file containing the algorithm output.
    OUTPUT_FILE = 'output'

    # The standard dimensions of the input file
    FILE_DIMS = 'Genes/Conditions'

    params = {
        'thresh': 0.8,
        'bases': 1000,
        'overlap': 100.0,
        'out_format': 0
    }

    def __init__(self, model='ccs', file_format='txt', temp=False, **kwargs):

        super().__init__(model, file_format, temp)

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, sep='\t', **kwargs):

        #_X = check_array(X)

        # Call to base method to set paths for temp dir and data.
        self.setup_io()
        # Creates file of input data according to algorithm requirements.
        self.format_input(X, sep=sep)
        # Call to application
        self.exec_clustering()

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        self.fetch_biclusters(X)

        #if self.temp:
        #    self.io_teardown_temp()

        #return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)

    def format_input(self, X, sep='\t', **kwargs):

        try:
            rows = kwargs['']
        except:
            rows = ['row{0}'.format(str(num)) for num in range(X.shape[0])]
        try:
            cols = kwargs['genes']
        except:
            cols = [
                '{0}col{1}'.format(sep, str(num)) for num in range(X.shape[1])
            ]
        with open(self.path_data, 'w') as outfile:
            outfile.write(self.FILE_DIMS)
            outfile.write(('').join(cols))
            outfile.write('\n')
            for num, row in enumerate(X):
                outfile.write('{0}{1}'.format(rows[num], sep))
                outfile.write(
                    ('').join(['{0}{1}'.format(sep, str(val)) for val in row])
                )
                outfile.write('\n')
            outfile.close()

        return self

    def exec_clustering(self):

        # Create file holding results
        self._setup_exec()

        # Change to current working dir
        current_loc = os.getcwd()

        try:
            os.chdir(self.path_dir)
            # ccs -t 0.9 -i input_file -o output_file -m 50 -p 1 -g 100.0
            command = (
                '{model} -t {thresh} -i {infile} -o {outfile} '
                '-m {bases} -p {out_format} -g {overlap} {out_format}'
                ''.format(model=self.model, **self.params)
            )
            subprocess.check_call(command.split())

        except OSError:
            raise PathError('Executable {0} not on $PATH'.format(value))

        finally:
            os.chdir(current_loc)

    def _setup_exec(self):

        self.params['infile'] = os.path.abspath(self.path_data)
        self.params['outfile'] = os.path.join(
            os.path.abspath(self.path_dir),
            '{0}.{1}'.format(self.OUTPUT_FILE, self.file_format)
        )

    def fetch_biclusters(self, X):

        results_file = os.path.join(
            self.path_dir, '{0}.{1}'.format(self.OUTPUT_FILE, self.file_format)
        )
        with open(results_file, 'r') as infile:

            header = infile.readline().split()
            print(header)

    def format_output(self):

        pass



# ERROR: The algorithm determines
class CPB(BinaryBiclusteringBase):
    """A wrapper for the CPB binary algorithm.

    Detects biclusters based on row correlations.

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Initial bicluster
    INIT_BINARY = 'init_bicluster'

    # Stem of output files holding the biclustering results.
    STEM_OUTPUT_FILE = '.out'

    # Hyperparameters
    params = {
        'nclus': 2,
        'targetpcc': 0.9,
        'fixed_row': -1,
        'fixed_col': -1,
        'fixw': 0,
        'min_seed_rows': 3,
        'max_seed_rows': None,
        'targetpcc': 0.9,
        'fixw': 0
    }

    def __init__(self, model='cpb', file_format='txt', temp=False, **kwargs):

        super().__init__(model, file_format, temp)

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        #_X = check_array(X)

        # Call to base method to set paths for temp dir and data.
        self.setup_io()
        # Creates file of input data according to algorithm requirements.
        self.format_input(X)
        # Call to application
        self.exec_clustering()

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        self.fetch_biclusters(X)

        if self.temp:
            self.io_teardown_temp()

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)

    def format_input(self, X):

        self.params['nrows'], self.params['ncols'] = np.shape(X)
        with open(self.path_data, 'w') as outfile:

            outfile.write(
                '{0} {1}'.format(self.params['nrows'], self.params['ncols'])
            )
            for row in X:
                outfile.write('\n')
                # TODO: Convert map row data to str instead of list comp?
                outfile.write(' '.join([str(value) for value in row]))
            outfile.write('\n')
            outfile.close()

        return self

    def exec_clustering(self):

        # Create a file to hold the results.
        self.create_results_file()
        # Change to current working dir.
        current_loc = os.getcwd()
        try:
            os.chdir(self.path_dir)
            command = (
                '{model} {outfile} {initfile} 1 {targetpcc} {fixw}'.format(
                    model=self.model,
                    outfile=os.path.abspath(self.path_data),
                    **self.params
                )
            )
            subprocess.check_call(command.split())

        except OSError:
            raise PathError('CPB not on $PATH')

        finally:
            os.chdir(current_loc)

        return self

    def create_results_file(self):

        self.params['initfile'] = os.path.join(
            self.path_dir, 'initfile.{0}'.format(self.file_format)
        )
        self.params['init_binary'] = self.INIT_BINARY

        # Create initial bicluster file
        command = (
            '{init_binary} '
            '{nrows} {ncols} {nclus} '
            '{min_seed_rows} {max_seed_rows} {fixed_row} {fixed_col} '
            '{initfile}'.format(**self.params)
        )
        try:
            subprocess.check_call(command.split())
        except OSError:
            raise PathError('Executable {0} not on $PATH'
                            ''.format(self.params['init_binary']))

        return self

    def fetch_biclusters(self, X):
        """

        """

        results_dir = os.path.abspath(self.path_dir)
        dir_files = os.listdir(results_dir)

        num_biclusters, rows, cols = 0, [], []
        for _file in dir_files:

            _, stem = os.path.splitext(_file)
            if stem == self.STEM_OUTPUT_FILE:

                output_file = os.path.join(results_dir, _file)
                row_idx, col_idx = self.format_output(output_file, X)
                rows.append(row_idx), cols.append(col_idx)
                print(row_idx, col_idx)
                num_biclusters += 1

        # Create row and column bicluster indicators.
        row_clusters = np.zeros((num_biclusters, X.shape[0]), dtype=bool)
        col_clusters = np.zeros((num_biclusters, X.shape[1]), dtype=bool)

        for num in range(num_biclusters):

            #print(row_clusters[num].shape, rows[num].shape)
            #print(col_clusters[num].shape, rows[num].shape)
            #print('input')
            #print(np.shape(rows[num]), np.shape(cols[num]))
            #print('containers')
            #print(row_clusters[num].shape, col_clusters[num].shape)
            pass
            #row_clusters[num, :][rows[num]] = True
            #col_clusters[num, :][cols[num]] = True

        self.rows_, self.columns_ = row_clusters, col_clusters
        self.biclusters_ = (self.rows_, self.columns_)

        return self

    def format_output(self, filename, X):
        """Reads the bicluster in a single CPB output file.

        Args:
            filename (str):
            X (array-like):

        Returns:
            (tuple): The bicluster row and column indicators.

        """

        rows, cols = [], []
        with open(filename, 'r') as outfile:

            target = rows
            for line in outfile.readlines():
                print(line.split())
                if line[0] == 'R':
                    continue
                elif line[0] == 'C':
                    target = cols
                    continue
                else:
                    target.append(int(line.split()[0]))

            rows.sort(), cols.sort()

        return rows, cols
