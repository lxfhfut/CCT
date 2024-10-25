import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import sys
from readlif.reader import LifFile
import pathlib
from pandas import DataFrame
from skimage.segmentation import flood, flood_fill

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from skimage import color
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QWidget, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QIcon
import PyQt5
from PyQt5.QtCore import Qt
from ui_mainwindow import Ui_MainWindow
from img_lib import *
from skimage import morphology


class CellCounter(QMainWindow):
    original_img = None
    current_img = None
    blob_img = None

    img_width = 0
    img_height = 0
    qpmap_width = 0
    qpmap_height = 0

    num_dots = 0
    num_cells = 0

    blobs_list = []

    cell_label_img = None
    label_type_img = None
    avg_gray_img = None
    label_img = None

    img_object = ImageProcLib()
    open_file_location = '/'
    model = StarDist2D.from_pretrained('2D_paper_dsb2018')

    # GUI initialization
    def __init__(self, parent=None):
        # initializing QWidget Qt module
        super(CellCounter, self).__init__()
        QWidget.__init__(self, parent)
        self.setWindowIcon(PyQt5.QtGui.QIcon('cct.ico'))

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.showMaximized()

        # assigning functions to be called on all button clicked and slider events
        self.ui.openImageButton.clicked.connect(lambda: self.open_image())
        self.ui.saveImageButton.clicked.connect(lambda: self.save_image())
        self.ui.cntDotsBnt.clicked.connect(lambda: self.count_dots())
        self.ui.ablThreshSlider.valueChanged.connect(lambda: self.count_dots())
        self.ui.relThreshSlider.valueChanged.connect(lambda: self.count_dots())
        self.ui.numSigmaSlider.valueChanged.connect(lambda: self.count_dots())
        self.ui.minSigmaSpinBox.valueChanged.connect(lambda: self.count_dots())
        self.ui.maxSigmaSpinBox.valueChanged.connect(lambda: self.count_dots())

        self.ui.cntCellsBtn.clicked.connect(lambda: self.count_cells())
        self.ui.minIntensitySlider.valueChanged.connect(lambda: self.int_size_thresholding())
        self.ui.minSizeSlider.valueChanged.connect(lambda: self.int_size_thresholding())
        self.ui.inputBrowseBtn.clicked.connect(lambda: self.select_input_folder())
        self.ui.outputBrowseBtn.clicked.connect(lambda: self.select_output_folder())
        self.ui.batchProcBtn.clicked.connect(lambda: self.batch_processing())
        self.ui.lifBrowseBtn.clicked.connect(lambda: self.select_lif_path())
        self.ui.saveBrowseBtn.clicked.connect(lambda: self.select_save_dir())
        self.ui.convertBtn.clicked.connect(lambda: self.save_lif_imgs())
        self.ui.imageDisplayLabel.leftClicked.connect(lambda x, y: self.add_objs(x, y))
        self.ui.imageDisplayLabel.rightClicked.connect(lambda x, y: self.remove_objs(x, y))

    def add_objs(self, x, y):
        if (self.ui.dotCntRadioBtn.isChecked()) and (self.original_img is not None) and (len(self.blobs_list) != 0):
            x_img, y_img = self.label2img_coordinates(x, y)
            inside = False
            for blob in self.blobs_list:
                y, x, r, t = blob
                if (x-x_img)**2 + (y-y_img)**2 <= r**2:
                    inside = True
                    break
            if not inside:
                self.blobs_list.append(np.array([y_img, x_img, 8, 1]))
                self.ui.batchInfoLabel.setText("A dot at (" + str(x_img) + ", " + str(y_img) + ") has been added.")
            # image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            image = self.original_img.copy()
            image[:, :, 0] = self.blob_img
            self.visualize_blobs(image)
        elif (self.ui.cellCntRadioBtn.isChecked()) and \
                (self.original_img is not None) and \
                (self.label_img is not None):
            tol = self.ui.toleranceSlider.value()

            x_img, y_img = self.label2img_coordinates(x, y)

            # if a non-cell object (with zero label) is selected
            label_selected = self.label_img[y_img, x_img]
            intensity_img = self.original_img[:, :, 2]
            min_intensity = int(self.ui.minIntensitySlider.value() * 0.8)

            if label_selected == 0 and intensity_img[y_img, x_img] > min_intensity:
                mask = flood(intensity_img, (y_img, x_img), tolerance=tol)
                mask = morphology.binary_dilation(mask, morphology.disk(3))
                mask = morphology.remove_small_holes(mask, 128)
                self.label_type_img[mask] = 1
                self.label_img[mask] = np.max(self.label_img) + 1
                self.visualize_cells()

    def remove_objs(self, x, y):
        if (self.ui.dotCntRadioBtn.isChecked()) and (len(self.blobs_list) != 0):
            x_img, y_img = self.label2img_coordinates(x, y)
            for idx, blob in enumerate(self.blobs_list):
                y, x, r, t = blob
                if (x - x_img) ** 2 + (y - y_img) ** 2 <= r ** 2:
                    self.blobs_list.pop(idx)
                    self.ui.batchInfoLabel.setText("A dot at (" + str(x_img) + ", " + str(y_img) + ") has been removed.")
                    break
            # image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            image = self.original_img.copy()
            image[:, :, 0] = self.blob_img
            self.visualize_blobs(image)
        elif (self.ui.cellCntRadioBtn.isChecked()) and \
                (self.label_img is not None):

            x_img, y_img = self.label2img_coordinates(x, y)

            # if a cell object (with non-zero label) is selected
            label_selected = self.label_img[y_img, x_img]
            if label_selected > 0:
                self.label_type_img[self.label_img == label_selected] = 0
                self.label_img[self.label_img == label_selected] = 0
                self.ui.batchInfoLabel.setText("A cell at (" + str(x_img) + ", " + str(y_img) + ") has been removed.")
                self.visualize_cells()

    def count_dots(self):
        self.current_img = self.original_img.copy()
        min_sigma = self.ui.minSigmaSpinBox.value()
        max_sigma = self.ui.maxSigmaSpinBox.value()
        num_sigma = self.ui.numSigmaSlider.value()
        abl_thres = self.ui.ablThreshSlider.value() / 100.0
        rel_thres = self.ui.relThreshSlider.value() / 100.0
        if abl_thres == 0: abl_thres = None
        self.blob_img, blobs = self.img_object.count_dots(self.current_img,
                                                     min_sigma, max_sigma, num_sigma,
                                                     threshold=abl_thres, threshold_rel=rel_thres)

        blobs[:, 2] = blobs[:, 2] * 8
        blobs_expand = np.zeros((blobs.shape[0], 4), dtype=blobs.dtype)
        blobs_expand[:, :3] = blobs
        self.blobs_list = [blob for blob in blobs_expand]
        # image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        image = self.current_img.copy()
        image[:, :, 0] = self.blob_img
        # image[:, :, 1] = blob_img
        # image[:, :, 2] = blob_img
        self.visualize_blobs(image)

    def int_size_thresholding(self):
        if self.avg_gray_img is None:
            self.count_cells()
            return

        intensity_thres = self.ui.minIntensitySlider.value()
        size_thres = self.ui.minSizeSlider.value()

        mask = self.avg_gray_img >= intensity_thres
        mask = morphology.remove_small_objects(mask, size_thres)

        self.label_type_img = np.zeros_like(self.cell_label_img, dtype=np.int32)
        self.label_img = self.cell_label_img.copy()
        self.label_img[(1 - mask).astype(bool)] = 0
        self.visualize_cells()

    def count_cells(self):
        self.current_img = self.original_img.copy()
        prob_thres = self.ui.probThreshSlider.value() / 20.0
        nms_thres = self.ui.nmsThreshSlider.value() / 20.0

        self.cell_label_img = self.img_object.count_cells(self.model, self.current_img, prob_thres, nms_thres)
        avg_color_img = color.label2rgb(self.cell_label_img, self.current_img, kind='avg')
        self.avg_gray_img = cv2.cvtColor(avg_color_img, cv2.COLOR_RGB2GRAY)
        self.label_type_img = np.zeros_like(self.cell_label_img, dtype=np.int32)
        self.int_size_thresholding()

    def select_input_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Select Input Folder", directory=self.open_file_location)
        if name:
            self.ui.inputDirLineEdit.setText(name)
            self.open_file_location = name

    def select_lif_path(self):
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.DontUseNativeDialog
        select_lif_window = QFileDialog()  # opens a new Open Image dialog box
        lif_file_path = QFileDialog.getOpenFileName(select_lif_window, 'Select Lif File',
                                                    directory=self.open_file_location,
                                                    filter="Image files (*.lif)", **kwargs)

        # check if image path is not null or empty
        if lif_file_path[0] != '':
            self.open_file_location = os.path.dirname(lif_file_path[0])
            self.ui.inputLifLineEdit.setText(lif_file_path[0])
            if self.ui.saveDirLineEdit.text() != '':
                self.ui.convertBtn.setEnabled(True)

    def select_save_dir(self):
        name = QFileDialog.getExistingDirectory(self, "Select Save Directory", directory=self.open_file_location)
        if name:
            self.ui.saveDirLineEdit.setText(name)
            self.open_file_location = name
            if self.ui.inputLifLineEdit.text() != '':
                self.ui.convertBtn.setEnabled(True)

    def save_lif_imgs(self):
        lif_path = self.ui.inputLifLineEdit.text()
        dst_dir = self.ui.saveDirLineEdit.text()
        if lif_path == "" or dst_dir == "":
            self.ui.resultLabelMiddle.setText("Please select the input and output directory!")
            return
        lif_obj = LifFile(lif_path)
        name = os.path.basename(lif_path)[:-4]
        pathlib.Path(os.path.join(dst_dir, name, 'Dots')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(dst_dir, name, 'Cells')).mkdir(parents=True, exist_ok=True)
        for series_idx in range(lif_obj.num_images):
            self.ui.batchInfoLabel.setText(f"Processing series {series_idx + 1}/{lif_obj.num_images} ...")
            self.ui.batchInfoLabel.repaint()
            QApplication.processEvents()
            series = lif_obj.get_image(series_idx)
            series_name = series.name
            imgs = []
            if series.dims.z > 1:
                for r, g, b in zip(series.get_iter_z(c=0), series.get_iter_z(c=1), series.get_iter_z(c=2)):
                    imgs.append(np.dstack((np.asarray(r), np.asarray(g), np.asarray(b))))
            elif series.dims.t > 1:
                for r, g, b in zip(series.get_iter_t(c=0), series.get_iter_t(c=1), series.get_iter_t(c=2)):
                    imgs.append(np.dstack((np.asarray(r), np.asarray(g), np.asarray(b))))
            else:
                for c in series.get_iter_c(z=0, t=0, m=0):
                    imgs.append(np.asarray(c))

            cell_img = imgs[0]
            if cell_img.ndim == 3:
                for ch_idx in range(cell_img.shape[2]):
                    cell_img[:, :, ch_idx] = ((cell_img[:, :, ch_idx] - cell_img.min()) /
                                              (cell_img.max() - cell_img.min()) * 255).astype('uint8')
            else:
                cell_img = ((cell_img - cell_img.min()) / (cell_img.max() - cell_img.min()) * 255).astype('uint8')
                cell_img = np.tile(cell_img[:, :, None], [1, 1, 3])
                # cell_img[:, :, 2] = 0
                # cell_img[:, :, 1] = 0  # Convert cell images to blue images following opencv convention

            blob_img = imgs[1]
            if blob_img.ndim == 3:
                for ch_idx in range(blob_img.shape[2]):
                    blob_img[:, :, ch_idx] = ((blob_img[:, :, ch_idx] - blob_img.min()) /
                                            (blob_img.max() - blob_img.min()) * 255).astype('uint8')
            else:
                blob_img = ((blob_img - blob_img.min()) / (blob_img.max() - blob_img.min()) * 255).astype('uint8')
                blob_img = np.tile(blob_img[:, :, None], [1, 1, 3])

                # put cell image in blue channel as background to facilitate dots counting
                blob_img[:, :, 0] = cell_img[:, :, 0]
                blob_img[:, :, 1] = 0

            cv2.imwrite(
                os.path.join(dst_dir, name, 'Dots',
                             name + '_series_' + str(series_idx + 1) + '_' + series_name + '.png'), blob_img)
            cv2.imwrite(
                os.path.join(dst_dir, name, 'Cells',
                             name + '_series_' + str(series_idx + 1) + '_' + series_name + '.png'), cell_img)
        self.ui.batchInfoLabel.setText(f"Images have been saved to '{dst_dir}'.")

    def select_output_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Select Output Folder", directory=self.open_file_location)
        if name:
            self.ui.outputDirLineEdit.setText(name)
            self.open_file_location = name

    def batch_processing(self):
        def get_img_paths(input_dir):
            img_paths = []
            for f in os.listdir(input_dir):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in [".jpg", ".tif", ".png"]:
                    continue
                img_paths.append(os.path.join(input_dir, f))
            return img_paths

        input_dir = self.ui.inputDirLineEdit.text()
        output_dir = self.ui.outputDirLineEdit.text()
        if input_dir == "" or output_dir == "":
            self.ui.resultLabelMiddle.setText("Please select the input and output directory!")
            return

        img_paths = get_img_paths(input_dir)
        img_names = []
        counts = []
        for idx, img_path in enumerate(img_paths):
            self.ui.batchInfoLabel.setText(f"Processing image {idx+1}/{len(img_paths)} ...")
            self.ui.batchInfoLabel.repaint()
            QApplication.processEvents()
            img_names.append(os.path.basename(img_path))
            img = cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2RGB)
            self.original_img = img.copy()
            if self.ui.dotCntRadioBtn.isChecked():  # dots counting
                self.count_dots()
                cv2.imwrite(os.path.join(output_dir,
                                         os.path.basename(img_path)[:-4] + "_dots_" + str(self.num_dots) + ".png"),
                            cv2.cvtColor(self.current_img, cv2.COLOR_RGB2BGR))
                counts.append(self.num_dots)
            else:
                self.count_cells()
                cv2.imwrite(os.path.join(output_dir,
                                         os.path.basename(img_path)[:-4] + "_cells_" + str(self.num_cells) + ".png"),
                            cv2.cvtColor(self.current_img, cv2.COLOR_RGB2BGR))
                counts.append(self.num_cells)

        if self.ui.dotCntRadioBtn.isChecked():
             df = DataFrame({'Image name': img_names, 'Number of dots': counts})
             df.to_csv(os.path.join(output_dir, 'Result_dots.csv'), index=False)
        else:
             df = DataFrame({'Image name': img_names, 'Number of cells': counts})
             df.to_csv(os.path.join(output_dir, 'Result_cells.csv'), index=False)

        self.ui.batchInfoLabel.setText(f"Results have been saved to \n{output_dir}.")

    # called when Open button is clicked
    def open_image(self):
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.DontUseNativeDialog
        open_image_window = QFileDialog()  # opens a new Open Image dialog box
        image_path = QFileDialog.getOpenFileName(open_image_window, 'Open Image',
                                                 directory=self.open_file_location,
                                                 filter="Image files (*.png *.jpg *.tif)", **kwargs)

        # check if image path is not null or empty
        if image_path[0] != '':

            # initialize class variables
            self.current_img = None
            self.blobs_list = []
            self.qpmap_height = 0
            self.qpmap_width = 0

            # read image at selected path to a numpy ndarray object as color image
            path, _ = image_path
            self.open_file_location = os.path.dirname(path)
            self.current_img = cv2.imread(path, 1)

            # convert the image read to BGR format from default RGB format
            self.current_img = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)

            # set image specific class variables based on current image
            self.img_width = self.current_img.shape[1]
            self.img_height = self.current_img.shape[0]

            self.original_img = self.current_img.copy()

            self.display_image()

            self.enable_options()

    # called when Save button is clicked
    def save_image(self):
        # configure the save image dialog box to use .jpg extension for image if not provided in file name
        dialog = QFileDialog()
        dialog.setDefaultSuffix('jpg')
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        # open the save dialog box and wait until user clicks 'Save' button in the dialog box
        if dialog.exec_() == QDialog.Accepted:
            save_image_filename = dialog.selectedFiles()[0]  # select the first path as image save location

            # write current image to the file path selected by user
            cv2.imwrite(save_image_filename,
                        cv2.cvtColor(self.current_img, cv2.COLOR_RGB2BGR))

    # display_image converts current image from ndarray format to pixmap and assigns it to image display label
    def display_image(self):
        display_size = self.ui.imageDisplayLabel.size()  # setting display to the size of the image display label

        image = np.array(self.current_img.copy())  # copying current image to temporary variable for processing pixmap
        zero = np.array([0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            qImage = QImage(image, self.img_width, self.img_height,
                            self.img_width * 3, QImage.Format_RGB888)

            # converting QImage to QPixmap for loading in image display label
            pixmap = QPixmap()
            QPixmap.convertFromImage(pixmap, qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.qpmap_width = pixmap.width()
            self.qpmap_height = pixmap.height()
            self.ui.imageDisplayLabel.setPixmap(pixmap)  # set pixmap to image display label in GUI


    def visualize_blobs(self, image):
        min_sigma = self.ui.minSigmaSpinBox.value()
        max_sigma = self.ui.maxSigmaSpinBox.value()

        num_sigma = self.ui.numSigmaSlider.value()
        abl_thres = self.ui.ablThreshSlider.value() / 100.0
        rel_thres = self.ui.relThreshSlider.value() / 100.0
        for idx, blob in enumerate(self.blobs_list):
            y, x, r, t = blob
            y, x, r, t = int(y), int(x), int(r), int(t)
            if self.ui.dotRadiusCheckBox.isChecked():
                # for automatic results
                if t == 0:
                    image = cv2.circle(image, center=(x, y), radius=r, color=(255, 255, 0))
                else:  # for manual results
                    image = cv2.circle(image, center=(x, y), radius=r, color=(0, 255, 255))
            offset = r + 5
            coordinates = [(x + offset, y + offset),
                           (x - offset, y + offset),
                           (x - offset, y - offset),
                           (x + offset, y - offset)]
            if self.ui.dotNumberCheckBox.isChecked():
                for crd in coordinates:
                    if 0 <= crd[0] < self.img_width and 0 <= crd[1] < self.img_height:
                        image = cv2.putText(image, str(idx + 1), (crd[0], crd[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        break
        self.num_dots = len(self.blobs_list)
        self.current_img = image
        # self.ui.resultLabelMiddle.setText(
        #     'min sigma: {:.2f}\nmax sigma: {:.2f}\nnum_sigma: {}\nabl_thres: {}\nrel_thres: {}\n\nDots detected: {}'.format(
        #         min_sigma, max_sigma, num_sigma, abl_thres, rel_thres, self.num_dots))
        self.ui.resultLabelMiddle.setText(f'Dots detected: {self.num_dots}')
        self.display_image()

    def visualize_cells(self):
        intensity_thres = self.ui.minIntensitySlider.value()
        size_thres = self.ui.minSizeSlider.value()

        show_boundary = self.ui.cellBoundaryCheckBox.isChecked()
        show_number = self.ui.cellNumberCheckBox.isChecked()

        if show_boundary:
            # for automatic results
            img_with_bnd = mark_boundaries(self.original_img, self.label_img, (1, 1, 0), mode='thick',
                                           background_label=0)
            # for manual results
            img_with_bnd = mark_boundaries(img_with_bnd, self.label_type_img, (0, 1, 1), mode='thick',
                                           background_label=0)
            img_with_bnd = (img_with_bnd * 255).astype(np.uint8)
        else:
            img_with_bnd = self.original_img.copy()
        unique_labels = list(np.unique(self.label_img[:]))

        if show_number:
            for label_idx, label in enumerate(unique_labels):
                if label == 0:
                    continue
                coordinates = np.argwhere(self.label_img == label)

                center = np.max(coordinates, axis=0)
                if center[0] >= self.img_height:
                    center[0] = self.img_height - 1
                if center[1] >= self.img_width:
                    center[1] = self.img_width - 1
                center = center[::-1]
                img_with_bnd = cv2.putText(img_with_bnd, str(label_idx + 1),
                                           tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                                           0.5, (0, 255, 0), 1, cv2.LINE_AA)
        self.current_img = img_with_bnd
        self.num_cells = np.unique(self.label_img[:]).size

        prob_thres = self.ui.probThreshSlider.value() / 20.0
        nms_thres = self.ui.nmsThreshSlider.value() / 20.0
        # self.ui.resultLabelMiddle.setText(
        #     'prob_thres: {:.2f}\nnms_thres: {:.2f}\nmin_intensity: {}\nmin_size: {}\n\nCells detected: {}'.format(
        #         prob_thres, nms_thres, intensity_thres, size_thres, self.num_cells))
        self.ui.resultLabelMiddle.setText(f'Cells detected: {self.num_cells}')
        self.display_image()

    def label2img_coordinates(self, x, y):
        display_size = self.ui.imageDisplayLabel.size()
        label_width, label_height = display_size.width(), display_size.height()
        offset_x = (label_width - self.qpmap_width) // 2
        offset_y = (label_height - self.qpmap_height) // 2
        x_actual, y_actual = x - offset_x, y - offset_y
        x_actual = max(0, min(x_actual, self.qpmap_width - 1))
        y_actual = max(0, min(y_actual, self.qpmap_height - 1))
        return int(x_actual / self.qpmap_width * self.img_width), int(y_actual / self.qpmap_height * self.img_height)

    def enable_options(self):
        self.ui.cntDotsBnt.setEnabled(True)
        self.ui.saveImageButton.setEnabled(True)
        self.ui.dotCntRadioBtn.setEnabled(True)
        self.ui.minSigmaSpinBox.setEnabled(True)
        self.ui.maxSigmaSpinBox.setEnabled(True)
        self.ui.numSigmaSlider.setEnabled(True)
        self.ui.ablThreshSlider.setEnabled(True)
        self.ui.relThreshSlider.setEnabled(True)
        self.ui.cellCntRadioBtn.setEnabled(True)
        self.ui.dotRadiusCheckBox.setEnabled(True)
        self.ui.dotNumberCheckBox.setEnabled(True)
        self.ui.inputBrowseBtn.setEnabled(True)
        self.ui.outputBrowseBtn.setEnabled(True)
        self.ui.inputDirLineEdit.setEnabled(True)
        self.ui.outputDirLineEdit.setEnabled(True)
        self.ui.batchProcBtn.setEnabled(True)

        self.ui.probThreshSlider.setEnabled(False)
        self.ui.nmsThreshSlider.setEnabled(False)
        self.ui.minIntensitySlider.setEnabled(False)
        self.ui.minSizeSlider.setEnabled(False)
        self.ui.toleranceSlider.setEnabled(False)
        self.ui.cellBoundaryCheckBox.setEnabled(False)
        self.ui.cellNumberCheckBox.setEnabled(False)
        self.ui.cntCellsBtn.setEnabled(False)
        self.ui.dotCntRadioBtn.setChecked(True)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setWindowIcon(QIcon("./cct.ico"))
    application = CellCounter()
    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
