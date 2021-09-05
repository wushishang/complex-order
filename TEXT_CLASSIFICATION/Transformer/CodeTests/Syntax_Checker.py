import sys
import os
import subprocess
from run_experiments import SYNTAX_CHECK, TASKS
from config import Config
from tqdm import tqdm

PRJ_PATH = os.path.dirname(os.getcwd())
os.chdir(PRJ_PATH)

class SyntaxCheck:
    def __init__(self, force_from_scratch=False):
        """
        :param force_from_scratch: Force training from scratch?
        """

        assert SYNTAX_CHECK, "SYNTAX CHECK in run_experiments.py must be True"
        assert isinstance(TASKS, list), "TASKS must be a nonempty list"
        assert len(TASKS) > 0, "TASKS must be a nonempty list"

        self.tasks = TASKS
        self.force_from_scratch = force_from_scratch
        self.error_list = list()
        self.error_commands = list()

        if not os.path.exists('train.py'):
            raise RuntimeError("Must have train.py in directory")

    def run_tests(self):

        tasks = [(t[1].split("train.py ")[1].split(" "), t[3]) for t in self.tasks]
        for task in tqdm(tasks):
            command_list = list(filter(None, task[0]))
            checkpoint = self.get_checkpoint_path(command_list)

            # Modify the model ID if we want to train from scratch
            already_exists = os.path.exists(checkpoint)

            epo = 4  # change the epoch to get a new ID
            while self.force_from_scratch and already_exists:
                print("WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                epo += 1
                command_list += ['-ne', str(epo)]
                checkpoint = self.get_checkpoint_path(command_list)
                already_exists = os.path.exists(checkpoint)

            tqdm.write(f"Running {command_list}")
            try:
                output = subprocess.run(['python', 'train.py'] + command_list, capture_output=True)
            except:
                print("there was a problem in the subprocess syntax")
                print(sys.exc_info())
                sys.exit()

            if output.returncode != 0:
                tqdm.write("Error found")
                self.error_list.append(output.stderr.decode('utf-8'))
                self.error_commands.append(command_list)
                print(self.error_list[-1])
            else:
                tqdm.write("Success!")


    def open_output(self):
        self.fo = open("syntax_checker.txt", 'w')

    def close_output(self):
        self.fo.close()

    def get_checkpoint_path(self, command_list):
        cfg = Config(command_list)
        return cfg.checkpoint_file_name()

    def main(self):
        self.run_tests()
        print(f"There were {len(self.error_list)} errors")
        for err, cmd in zip(self.error_list, self.error_commands):
            print("------Error-------")
            print(" ".join(cmd))
            print(err)


if __name__ == "__main__":
    sc = SyntaxCheck()
    sc.main()
