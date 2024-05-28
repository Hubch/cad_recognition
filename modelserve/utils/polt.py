import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font('Arial.Unicode.ttf' if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, '9.2.0'):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
        else:  # use cv2
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # font scale
        # Pose
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                # NOTE: PIL-version polygon needs tuple type.
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = p1[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                # NOTE: cv2-version polylines needs np.asarray type.
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.sf,
                            txt_color,
                            thickness=self.tf,
                            lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top', box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if '\n' in text:
                lines = text.split('\n')
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                outside = xy[1] - h >= 3
                p2 = xy[0] + w, xy[1] - h - 3 if outside else xy[1] + h + 3
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    # Object Counting Annotator
    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line
        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

    def count_labels(self, in_count=0, out_count=0, count_txt_size=2, color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        Plot counts for object counter
        Args:
            in_count (int): in count value
            out_count (int): out count value
            count_txt_size (int): text size for counts display
            color (tuple): background color of counts display
            txt_color (tuple): text color of counts display
        """
        self.tf = count_txt_size
        tl = self.tf or round(0.002 * (self.im.shape[0] + self.im.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)
        gap = int(24 * tl)  # gap between in_count and out_count based on line_thickness

        # Get text size for in_count and out_count
        t_size_in = cv2.getTextSize(str(in_count), 0, fontScale=tl / 2, thickness=tf)[0]
        t_size_out = cv2.getTextSize(str(out_count), 0, fontScale=tl / 2, thickness=tf)[0]

        # Calculate positions for in_count and out_count labels
        text_width = max(t_size_in[0], t_size_out[0])
        text_x1 = (self.im.shape[1] - text_width - 120 * self.tf) // 2 - gap
        text_x2 = (self.im.shape[1] - text_width + 120 * self.tf) // 2 + gap
        text_y = max(t_size_in[1], t_size_out[1])

        # Create a rounded rectangle for in_count
        cv2.rectangle(self.im, (text_x1 - 5, text_y - 5), (text_x1 + text_width + 7, text_y + t_size_in[1] + 7), color,
                      -1)
        cv2.putText(self.im,
                    str(in_count), (text_x1, text_y + t_size_in[1]),
                    0,
                    tl / 2,
                    txt_color,
                    self.tf,
                    lineType=cv2.LINE_AA)

        # Create a rounded rectangle for out_count
        cv2.rectangle(self.im, (text_x2 - 5, text_y - 5), (text_x2 + text_width + 7, text_y + t_size_out[1] + 7), color,
                      -1)
        cv2.putText(self.im,
                    str(out_count), (text_x2, text_y + t_size_out[1]),
                    0,
                    tl / 2,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA)

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """Calculate the pose angle for object
        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c
        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_points(self, keypoints, indices=[2, 5, 7], shape=(640, 640), radius=2):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): list of keypoints data to be plotted
            indices (list): keypoints ids list to be plotted
            shape (tuple): imgsz for model inference
            radius (int): Keypoint radius value
        """
        nkpts, ndim = keypoints.shape
        nkpts == 17 and ndim == 3
        for i, k in enumerate(keypoints):
            if i in indices:
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return self.im

    def plot_angle_and_count_and_stage(self, angle_text, count_text, stage_text, center_kpt, line_thickness=2):
        """
        Plot the pose angle, count value and step stage.

        Args:
            angle_text (str): angle value for workout monitoring
            count_text (str): counts value for workout monitoring
            stage_text (str): stage decision for workout monitoring
            center_kpt (int): centroid pose index for workout monitoring
            line_thickness (int): thickness for text display
        """
        angle_text, count_text, stage_text = (f' {angle_text:.2f}', 'Steps : ' + f'{count_text}', f' {stage_text}')
        font_scale = 0.6 + (line_thickness / 10.0)

        # Draw angle
        (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 0, font_scale, line_thickness)
        angle_text_position = (int(center_kpt[0]), int(center_kpt[1]))
        angle_background_position = (angle_text_position[0], angle_text_position[1] - angle_text_height - 5)
        angle_background_size = (angle_text_width + 2 * 5, angle_text_height + 2 * 5 + (line_thickness * 2))
        cv2.rectangle(self.im, angle_background_position, (angle_background_position[0] + angle_background_size[0],
                                                           angle_background_position[1] + angle_background_size[1]),
                      (255, 255, 255), -1)
        cv2.putText(self.im, angle_text, angle_text_position, 0, font_scale, (0, 0, 0), line_thickness)

        # Draw Counts
        (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, 0, font_scale, line_thickness)
        count_text_position = (angle_text_position[0], angle_text_position[1] + angle_text_height + 20)
        count_background_position = (angle_background_position[0],
                                     angle_background_position[1] + angle_background_size[1] + 5)
        count_background_size = (count_text_width + 10, count_text_height + 10 + (line_thickness * 2))

        cv2.rectangle(self.im, count_background_position, (count_background_position[0] + count_background_size[0],
                                                           count_background_position[1] + count_background_size[1]),
                      (255, 255, 255), -1)
        cv2.putText(self.im, count_text, count_text_position, 0, font_scale, (0, 0, 0), line_thickness)

        # Draw Stage
        (stage_text_width, stage_text_height), _ = cv2.getTextSize(stage_text, 0, font_scale, line_thickness)
        stage_text_position = (int(center_kpt[0]), int(center_kpt[1]) + angle_text_height + count_text_height + 40)
        stage_background_position = (stage_text_position[0], stage_text_position[1] - stage_text_height - 5)
        stage_background_size = (stage_text_width + 10, stage_text_height + 10)

        cv2.rectangle(self.im, stage_background_position, (stage_background_position[0] + stage_background_size[0],
                                                           stage_background_position[1] + stage_background_size[1]),
                      (255, 255, 255), -1)
        cv2.putText(self.im, stage_text, stage_text_position, 0, font_scale, (0, 0, 0), line_thickness)

    def seg_bbox(self, mask, mask_color=(255, 0, 255), det_label=None, track_label=None):
        """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (list): masks data list for instance segmentation area plotting
            mask_color (tuple): mask foreground color
            det_label (str): Detection label text
            track_label (str): Tracking label text
        """
        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)

        label = f'Track ID: {track_label}' if track_label else det_label
        text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)

        cv2.rectangle(self.im, (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
                      (int(mask[0][0]) + text_size[0] // 2 + 5, int(mask[0][1] + 5)), mask_color, -1)

        cv2.putText(self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, (255, 255, 255),
                    2)

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255), thickness=2, pins_radius=10):
        """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
            thickness (int): int value for line thickness
            pins_radius (int): visioneye point radius value
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, pins_radius, pin_color, -1)
        cv2.circle(self.im, center_bbox, pins_radius, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, thickness)
