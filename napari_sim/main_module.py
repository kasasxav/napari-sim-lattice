from magicgui import magic_factory, widgets
import napari
from napari_sim.reconstruction import Reconstruction
from napari.qt import thread_worker
from concurrent.futures import Future


@magic_factory
def sim_widget(
    image: napari.types.ImageData, pixel_size=65, periodicity=312, na=1.4, wvl=0.51, pad_px=2, offset=100, w=0.01, rad=0.7, cd=1, t=1
) -> Future[napari.types.ImageData]:

    future: Future[napari.types.ImageData] = Future()
    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    sim_widget.insert(0, pbar)  # add progress bar to the top of widget

    def _on_done(result, self=sim_widget):
        future.set_result(result)
        self.remove(pbar)

    @thread_worker()
    def _reconstruct():
        rec = Reconstruction(pixel_size, periodicity, na, wvl, pad_px, offset, w,
                 rad, cd, t)
        sr = rec.run(image)
        return sr

    worker = _reconstruct()
    worker.returned.connect(_on_done)
    worker.start()

    return future


@magic_factory
def wf_widget(image: napari.types.ImageData) -> napari.types.ImageData:
    rec = Reconstruction()
    return rec.get_wf(image)
