import pickle
import os
from time import time
import matplotlib.pyplot as plt

from utils import *
import pdb


class Camera(object):
    def __init__(self, R, t, ref):
        self.R = R
        self.t = t
        self.ref = ref


class Match(object):
    def __init__(self, matches, img1pts, img2pts, img1idx, img2idx, mask):
        self.matches = matches
        self.img1pts, self.img2pts = img1pts, img2pts
        self.img1idx, self.img2idx = img1idx, img2idx
        self.mask = mask


class SFM(object):
    def __init__(self):

        self.point_cloud = np.zeros((0, 3))

        # setting up directory stuff..
        self.out_dir = "results"
        self.dataset = "fountain-P11"
        self.images_dir = os.path.join('data/fountain-P11/images')
        self.feat_dir = os.path.join("data/fountain-P11/features/SIFT")
        self.matches_dir = os.path.join("data/fountain-P11/matches/BFMatcher")
        self.out_cloud_dir = os.path.join(self.out_dir, self.dataset, 'point-clouds')
        self.out_err_dir = os.path.join(self.out_dir, self.dataset, 'errors')
        self.plot_error = False
        self.ext = 'png,jpg'
        self.matcher = 'BFMatcher'
        self.pnp_method = 'SOLVEPNP_DLS'
        self.pnp_prob = 0.99
        self.cross_check = True
        self.calibration_mat = 'benchmark'
        self.fund_method = cv2.FM_RANSAC
        self.outlier_thres = 0.9
        self.fund_prob = 0.5
        self.reprojection_thres = 8.0

        # output directories
        if not os.path.exists(self.out_cloud_dir):
            os.makedirs(self.out_cloud_dir)

        if (self.plot_error is True) and (not os.path.exists(self.out_err_dir)):
            os.makedirs(self.out_err_dir)

        self.image_names = [x.split('.')[0] for x in sorted(os.listdir(self.images_dir))
                            if x.split('.')[-1] in self.ext]

        # setting up shared parameters for the pipeline
        self.image_data, self.matches_data, errors = {}, {}, {}
        self.matcher = getattr(cv2, self.matcher)(crossCheck=self.cross_check)

        if self.calibration_mat == 'benchmark':
            self.K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
        elif self.calibration_mat == 'lg_g3':
            self.K = np.array([[3.97 * 320, 0, 320], [0, 3.97 * 320, 240], [0, 0, 1]])
        else:
            raise NotImplementedError

    def _LoadFeatures(self, name):
        with open(os.path.join(self.feat_dir, 'kp_{}.pkl'.format(name)), 'rb') as f:
            kp = pickle.load(f, encoding='UTF-8')
        kp = DeserializeKeypoints(kp)

        with open(os.path.join(self.feat_dir, 'desc_{}.pkl'.format(name)), 'rb') as f:
            desc = pickle.load(f)

        return kp, desc

    def _LoadMatches(self, name1, name2):
        with open(os.path.join(self.matches_dir, 'match_{}_{}.pkl'.format(name1, name2)), 'rb') as f:
            matches = pickle.load(f)
        matches = DeserializeMatches(matches)
        return matches

    def _GetAlignedMatches(self, kp1, kp2, matches):
        img1idx = np.array([m.queryIdx for m in matches])
        img2idx = np.array([m.trainIdx for m in matches])

        # filtering out the keypoints that were matched.
        kp1_ = (np.array(kp1))[img1idx]
        kp2_ = (np.array(kp2))[img2idx]

        # retreiving the image coordinates of matched keypoints
        img1pts = np.array([kp.pt for kp in kp1_])
        img2pts = np.array([kp.pt for kp in kp2_])

        return img1pts, img2pts, img1idx, img2idx

    def _BaselinePoseEstimation(self, name1, name2):

        kp1, desc1 = self._LoadFeatures(name1)
        kp2, desc2 = self._LoadFeatures(name2)

        matches = self._LoadMatches(name1, name2)
        matches = sorted(matches, key=lambda x: x.distance)

        img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1, kp2, matches)

        F, mask = cv2.findFundamentalMat(img1pts, img2pts, method=self.fund_method, ransacReprojThreshold=self.outlier_thres)
        mask = mask.astype(bool).flatten()

        E = self.K.T.dot(F.dot(self.K))
        _, R, t, _ = cv2.recoverPose(E, img1pts[mask], img2pts[mask], self.K)

        self.image_data[name1] = [np.eye(3, 3), np.zeros((3, 1)), np.ones((len(kp1),)) * -1]
        self.image_data[name2] = [R, t, np.ones((len(kp2),)) * -1]

        self.matches_data[(name1, name2)] = [matches, img1pts[mask], img2pts[mask],
                                             img1idx[mask], img2idx[mask]]

        return R, t

    def _TriangulateTwoViews(self, name1, name2):

        def __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2):
            img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:, 0, :]
            img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:, 0, :]

            img1ptsNorm = (np.linalg.inv(self.K).dot(img1ptsHom.T)).T
            img2ptsNorm = (np.linalg.inv(self.K).dot(img2ptsHom.T)).T

            img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:, 0, :]
            img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:, 0, :]

            pts4d = cv2.triangulatePoints(np.hstack((R1, t1)), np.hstack((R2, t2)),
                                          img1ptsNorm.T, img2ptsNorm.T)
            pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:, 0, :]

            return pts3d

        def _Update3DReference(ref1, ref2, img1idx, img2idx, upp_limit, low_limit=0):
            ref1[img1idx] = np.arange(upp_limit) + low_limit
            ref2[img2idx] = np.arange(upp_limit) + low_limit

            return ref1, ref2

        R1, t1, ref1 = self.image_data[name1]
        R2, t2, ref2 = self.image_data[name2]

        _, img1pts, img2pts, img1idx, img2idx = self.matches_data[(name1, name2)]

        new_point_cloud = __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2)
        self.point_cloud = np.concatenate((self.point_cloud, new_point_cloud), axis=0)

        ref1, ref2 = _Update3DReference(ref1, ref2, img1idx, img2idx, new_point_cloud.shape[0],
                                        self.point_cloud.shape[0] - new_point_cloud.shape[0])
        self.image_data[name1][-1] = ref1
        self.image_data[name2][-1] = ref2

    def _TriangulateNewView(self, name):

        for prev_name in self.image_data.keys():
            if prev_name != name:
                kp1, desc1 = self._LoadFeatures(prev_name)
                kp2, desc2 = self._LoadFeatures(name)

                prev_name_ref = self.image_data[prev_name][-1]
                matches = self._LoadMatches(prev_name, name)
                matches = [match for match in matches if prev_name_ref[match.queryIdx] < 0]

                if len(matches) > 0:
                    matches = sorted(matches, key=lambda x: x.distance)

                    img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1, kp2,
                                                                                 matches)

                    F, mask = cv2.findFundamentalMat(img1pts, img2pts, method=self.fund_method, )
                    mask = mask.astype(bool).flatten()

                    self.matches_data[(prev_name, name)] = [matches, img1pts[mask], img2pts[mask],
                                                            img1idx[mask], img2idx[mask]]
                    self._TriangulateTwoViews(prev_name, name)

                else:
                    print('skipping {} and {}'.format(prev_name, name))

    def _NewViewPoseEstimation(self, name):

        def _Find2D3DMatches():

            matcher_temp = cv2.BFMatcher()
            kps, descs = [], []
            for n in self.image_names:
                if n in self.image_data.keys():
                    kp, desc = self._LoadFeatures(n)

                    kps.append(kp)
                    descs.append(desc)

            matcher_temp.add(descs)
            matcher_temp.train()

            kp, desc = self._LoadFeatures(name)

            matches_2d3d = matcher_temp.match(queryDescriptors=desc)

            # retrieving 2d and 3d points
            pts3d, pts2d = np.zeros((0, 3)), np.zeros((0, 2))
            for m in matches_2d3d:
                train_img_idx, desc_idx, new_img_idx = m.imgIdx, m.trainIdx, m.queryIdx
                point_cloud_idx = self.image_data[self.image_names[train_img_idx]][-1][desc_idx]

                # if the match corresponds to a point in 3d point cloud
                if point_cloud_idx >= 0:
                    new_pt = self.point_cloud[int(point_cloud_idx)]
                    pts3d = np.concatenate((pts3d, new_pt[np.newaxis]), axis=0)

                    new_pt = np.array(kp[int(new_img_idx)].pt)
                    pts2d = np.concatenate((pts2d, new_pt[np.newaxis]), axis=0)

            return pts3d, pts2d, len(kp)

        def __Find2D3DMatches():
            pts3d, pts2d = np.zeros((0, 3)), np.zeros((0, 2))
            kp, desc = self._LoadFeatures(name)

            i = 0

            while i < len(self.image_names):
                curr_name = self.image_names[i]

                if curr_name in self.image_data.keys():
                    matches = self._LoadMatches(curr_name, name)

                    ref = self.image_data[curr_name][-1]
                    pts3d_idx = np.array([ref[m.queryIdx] for m in matches \
                                          if ref[m.queryIdx] > 0])
                    pts2d_ = np.array([kp[m.trainIdx].pt for m in matches \
                                       if ref[m.queryIdx] > 0])

                    pts3d = np.concatenate((pts3d, self.point_cloud[pts3d_idx.astype(int)]), axis=0)
                    pts2d = np.concatenate((pts2d, pts2d_), axis=0)

                i += 1

            return pts3d, pts2d, len(kp)

        pts3d, pts2d, ref_len = _Find2D3DMatches()
        _, R, t, _ = cv2.solvePnPRansac(pts3d[:, np.newaxis], pts2d[:, np.newaxis], self.K, None,
                                        confidence=self.pnp_prob, flags=getattr(cv2, self.pnp_method),
                                        reprojectionError=self.reprojection_thres)
        R, _ = cv2.Rodrigues(R)
        self.image_data[name] = [R, t, np.ones((ref_len,)) * -1]

    def ToPly(self, filename):

        def _GetColors():
            colors = np.zeros_like(self.point_cloud)

            for k in self.image_data.keys():
                _, _, ref = self.image_data[k]
                kp, desc = self._LoadFeatures(k)
                kp = np.array(kp)[ref >= 0]
                image_pts = np.array([_kp.pt for _kp in kp])

                image = cv2.imread(os.path.join(self.images_dir, k + '.jpg'))[:, :, ::-1]

                colors[ref[ref >= 0].astype(int)] = image[image_pts[:, 1].astype(int),
                                                          image_pts[:, 0].astype(int)]

            return colors

        colors = _GetColors()
        pts2ply(self.point_cloud, colors, filename)

    def _ComputeReprojectionError(self, name):

        def _ComputeReprojections(X, R, t, K):
            outh = K.dot(R.dot(X.T) + t)
            out = cv2.convertPointsFromHomogeneous(outh.T)[:, 0, :]
            return out

        R, t, ref = self.image_data[name]
        reproj_pts = _ComputeReprojections(self.point_cloud[ref[ref > 0].astype(int)], R, t, self.K)

        kp, desc = self._LoadFeatures(name)
        img_pts = np.array([kp_.pt for i, kp_ in enumerate(kp) if ref[i] > 0])

        err = np.mean(np.sqrt(np.sum((img_pts - reproj_pts) ** 2, axis=-1)))

        if self.plot_error:
            fig, ax = plt.subplots()
            image = cv2.imread(os.path.join(self.images_dir, name + '.jpg'))[:, :, ::-1]
            ax = DrawCorrespondences(image, img_pts, reproj_pts, ax)

            ax.set_title('reprojection error = {}'.format(err))

            fig.savefig(os.path.join(self.out_err_dir, '{}.png'.format(name)))
            plt.close(fig)

        return err

    def Run(self):
        name1, name2 = self.image_names[0], self.image_names[1]

        total_time, errors = 0, []

        t1 = time()
        self._BaselinePoseEstimation(name1, name2)
        t2 = time()
        this_time = t2 - t1
        total_time += this_time
        print('Baseline Cameras {0}, {1}: Pose Estimation [time={2:.3}s]'.format(name1, name2, this_time))

        self._TriangulateTwoViews(name1, name2)
        t1 = time()
        this_time = t1 - t2
        total_time += this_time
        print
        'Baseline Cameras {0}, {1}: Baseline Triangulation [time={2:.3}s]'.format(name1,
                                                                                  name2, this_time)

        views_done = 2

        # 3d point cloud generation and reprojection error evaluation
        self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

        err1 = self._ComputeReprojectionError(name1)
        err2 = self._ComputeReprojectionError(name2)
        errors.append(err1)
        errors.append(err2)

        print('Camera {}: Reprojection Error = {}'.format(name1, err1))
        print('Camera {}: Reprojection Error = {}'.format(name2, err2))

        for new_name in self.image_names[2:]:
            # new camera registration
            t1 = time()
            self._NewViewPoseEstimation(new_name)
            t2 = time()
            this_time = t2 - t1
            total_time += this_time
            print('Camera {0}: Pose Estimation [time={1:.3}s]'.format(new_name, this_time))

            # triangulation for new registered camera
            self._TriangulateNewView(new_name)
            t1 = time()
            this_time = t1 - t2
            total_time += this_time
            print('Camera {0}: Triangulation [time={1:.3}s]'.format(new_name, this_time))

            # 3d point cloud update and error for new camera
            views_done += 1
            self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

            new_err = self._ComputeReprojectionError(new_name)
            errors.append(new_err)
            print('Camera {}: Reprojection Error = {}'.format(new_name, new_err))

        mean_error = sum(errors) / float(len(errors))
        print('Reconstruction Completed: Mean Reprojection Error = {2} [t={0:.6}s], Results stored in {1}'.format(total_time, self.out_dir, mean_error))


if __name__ == '__main__':
    print('start')
    sfm = SFM()
    sfm.Run()
    print('end')
