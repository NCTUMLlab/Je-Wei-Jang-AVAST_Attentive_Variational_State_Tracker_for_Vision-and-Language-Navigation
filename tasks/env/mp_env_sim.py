import MatterSim
import numpy as np


class MatterEnvSim():
    sim = None
    gif = None

    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__()
        self.config = config

        # rendering init
        self.width = config['r2r_env']['mp']['image_w']
        self.height = config['r2r_env']['mp']['image_h']
        self.vfov = np.deg2rad(config['r2r_env']['mp']['vfov'])
        self.hfov = self.vfov * self.width / self.height
        self.nav_text_color = [230, 40, 40]
        self.goal_text_color = [40, 40, 230]
        self.verbose = config['args']['verbose']
        self.rendering_idx = config['args']['rendering_idx']
        return

    def _sim_init(
        self,
        parallel_num: int
    ) -> None:
        self.sim = MatterSim.Simulator()
        self.sim.setDatasetPath(self.config['r2r_env']['mp']['skybox_dir'])
        self.sim.setNavGraphPath(self.config['r2r_env']['mp']['connectivity'])
        self.sim.setCameraResolution(self.config['r2r_env']['mp']['image_w'], self.config['mp']['image_h'])
        self.sim.setCameraVFOV(np.deg2rad(self.config['r2r_env']['mp']['vfov']))
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(parallel_num)
        self.sim.setCacheSize(2 * parallel_num)
        self.sim.setDepthEnabled(False)
        self.sim.setRenderingEnabled(True)
        self.sim.initialize()
        return

    def get_states(
        self
    ) -> list:
        return self.sim.getState()

    def new_episodes(
        self,
        scan_ids: list,
        vp_ids: list,
        headings: list,
        gen_gif: bool
    ) -> None:
        self._sim_init(len(scan_ids))
        self.sim.newEpisode(scan_ids, vp_ids, headings, [0] * len(scan_ids))
        if gen_gif:
            # init gif (batch, time_step, height, width, channel)
            self.gif = np.zeros(
                (
                    1,
                    int(self.config['r2r_env']['max_iteration'] * self.config['r2r_env']['action_space']),
                    self.config['r2r_env']['mp']['image_h'],
                    self.config['r2r_env']['mp']['image_w'],
                    3
                ),
                dtype=np.float32
            )
            self._add_frame_into_gif(self.sim.getState()[0])
        return

    def make_actions(
        self,
        forwards: list,
        headings: list,
        elevations: list
    ) -> None:
        self.sim.makeAction(forwards, headings, elevations)
        if not isinstance(self.gif, type(None)):
            self._add_frame_into_gif(self.sim.getState()[0])
        return

    def _add_frame_into_gif(
        self,
        state_info: MatterSim.SimState
    ) -> None:
        self.gif[self.rendering_idx, state_info.step] = np.array(
            state_info.rgb,
            copy=True,
            dtype=np.float32
        ) / 255
        return


def main():
    return


if __name__ == '__main__':
    main()
