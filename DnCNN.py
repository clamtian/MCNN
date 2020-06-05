import warnings
warnings.filterwarnings('ignore', category=Warning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from network import *
from PIL import Image
import scipy.misc as misc
import shelve
from skimage.measure import compare_psnr, compare_ssim
import pandas as pd

class DnCNN:
    def __init__(self):
        self.clean_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.noised_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.train_phase = tf.placeholder(tf.bool)
        dncnn = net("DnCNN")
        self.res = dncnn(self.noised_img, self.train_phase)
        self.denoised_img = self.noised_img - self.res
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.res - (self.noised_img - self.clean_img)), [1, 2, 3]))
        #优化算法（学习率）
        self.Opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        filepath = "./TrainingSet/"
        filenames = os.listdir(filepath)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state("./save_para")

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, "./save_para/DnCNN.ckpt")
        dbase = shelve.open("./save_para/mydbase")
        last_step = 0
        last_epoch = 0
        if "step" in dbase.keys():
            last_step = dbase['step']
        if "epoch" in dbase.keys():
            last_epoch = dbase['epoch']
        dbase.close()
        for epoch in range(last_epoch, 50):
            for i in range(last_step, filenames.__len__() // BATCH_SIZE):
                # print("i = " + str(i))
                cleaned_batch = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C])
                for idx, filename in enumerate(filenames[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]):
                    cleaned_batch[idx, :, :, 0] = np.array(Image.open(filepath + filename))
                noised_batch = cleaned_batch + np.random.normal(0, SIGMA, cleaned_batch.shape)
                self.sess.run(self.Opt, feed_dict={
                    self.clean_img: cleaned_batch,
                    self.noised_img: noised_batch,
                    self.train_phase: True})
                if i % 10 == 0:
                    [loss, denoised_img] = self.sess.run([self.loss, self.denoised_img],
                                                         feed_dict={self.clean_img: cleaned_batch,
                                                                    self.noised_img: noised_batch,
                                                                    self.train_phase: False})
                    print("Epoch: %d, Step: %d, Loss: %g" % (epoch, i, loss))
                    # compared = np.concatenate(
                    #     (cleaned_batch[0, :, :, 0], noised_batch[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
                    # Image.fromarray(np.uint8(compared)).save("./TrainingResults//" + str(epoch) + "_" + str(i) + ".jpg")
                if i % 100 == 0:
                    saver.save(self.sess, "./save_para/DnCNN.ckpt")
                    dbase = shelve.open("./save_para/mydbase")
                    dbase['step'] = i + 1
                    dbase.close()
                    print("Epoch: %d, Step: %d , data has been saved." % (epoch, i))
            np.random.shuffle(filenames)
            saver.save(self.sess, "./save_para/DnCNN.ckpt")
            dbase = shelve.open("./save_para/mydbase")
            dbase['epoch'] = epoch + 1
            dbase['step'] = last_step = 0
            dbase.close()
            print("Epoch %d has been finished." % epoch)


    # # def test(self, cleaned_path="./TestingSet//02.png"):
    # def test(self, path, filename):
    #     cleaned_path = path + filename
    #     saver = tf.train.Saver()
    #     saver.restore(self.sess, "./save_para/DnCNN.ckpt")
    #
    #
    #     # cleaned_img = np.reshape(np.array(misc.imresize(np.array(Image.open(cleaned_path)), [256, 256])), [1, 256, 256, 1])
    #     cleaned_img = np.reshape(np.array(misc.imresize(np.array(Image.open(cleaned_path)), [512, 512])),
    #                              [1, 512, 512, 1])
    #
    #     cleaned_img = np.reshape(np.array(Image.open(cleaned_path)), [1, 481, 321, 1])
    #
    #     noised = np.random.normal(0, SIGMA, cleaned_img.shape)
    #     noised_img = cleaned_img + noised
    #
    #     [denoised_img] = self.sess.run([self.denoised_img], feed_dict={
    #         self.clean_img: cleaned_img,
    #         self.noised_img: noised_img,
    #         self.train_phase: False})
    #
    #     compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
    #     # Image.fromarray(np.uint8(compared)).show()
    #
    #     # compared = np.concatenate(
    #     #     (cleaned_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
    #     # Image.fromarray(np.uint8(denoised_img.reshape((481, 321)))).save("./TestingResults/" + filename)
    #     #
    #     # cleaned_img = cleaned_img.reshape((481, 321))
    #     # denoised_img = denoised_img.reshape((481, 321))
    #     Image.fromarray(np.uint8(denoised_img.reshape((256, 256)))).save("./TestingResults/" + filename)
    #
    #     cleaned_img = cleaned_img.reshape((256, 256))
    #     denoised_img = denoised_img.reshape((256, 256))
    #
    #     # Image.fromarray(np.uint8(denoised_img.reshape((512, 512)))).save("./TestingResults/Set5/" + filename)
    #     #
    #     # cleaned_img = cleaned_img.reshape((512, 512))
    #     # denoised_img = denoised_img.reshape((512, 512))
    #
    #
    #     psnr = compare_psnr(cleaned_img, denoised_img)
    #     ssim = compare_ssim(cleaned_img, denoised_img)
    #     return psnr, ssim


    def test(self, path, path_, filename):
        cleaned_path = path + filename
        saver = tf.train.Saver()
        saver.restore(self.sess, "./save_para/DnCNN.ckpt")
        init_img = np.array(Image.open(cleaned_path))
        cleaned_img = np.reshape(init_img, [1, init_img.shape[0], init_img.shape[1], 1])

        noised = np.random.normal(0, SIGMA, cleaned_img.shape)
        noised_img = cleaned_img + noised

        [loss, denoised_img] = self.sess.run([self.loss, self.denoised_img], feed_dict={
            self.clean_img: cleaned_img,
            self.noised_img: noised_img,
            self.train_phase: False})
        compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
        Image.fromarray(np.uint8(noised_img.reshape((init_img.shape[0], init_img.shape[1])))).save("./TestingResults/" + path_ + "noised/" + filename)
        Image.fromarray(np.uint8(compared.reshape((compared.shape[0], compared.shape[1])))).save("./TestingResults/" + path_ + "compared/" + filename)
        Image.fromarray(np.uint8(denoised_img.reshape((init_img.shape[0], init_img.shape[1])))).save("./TestingResults/" + path_ + "denoised/" + filename)

        cleaned_img = cleaned_img.reshape((init_img.shape[0], init_img.shape[1]))
        denoised_img = denoised_img.reshape((init_img.shape[0], init_img.shape[1]))

        psnr = compare_psnr(cleaned_img, denoised_img)
        ssim = compare_ssim(cleaned_img, denoised_img)
        return psnr, ssim, loss

    def img_test(self, path_, epoch):
        path = "./TestingSet/" + path_  # 待读取的文件夹

        path_list = os.listdir(path)
        path_list.sort()  # 对读取的路径进行排序
        psnr_list = []
        ssim_list = []
        loss_list = []
        for filename in path_list:
            psnr, ssim , loss = dncnn.test(path, path_, filename)
            print(psnr, ssim, loss)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            loss_list.append(loss)
        dataframe = pd.DataFrame({'loss_value': loss_list})
        dataframe.to_csv("./test_save/" + path_ + "test_loss_save/epoch" + str(epoch) + ".csv", index=False, sep=',')
        dataframe = pd.DataFrame({'ssim_value': ssim_list})
        dataframe.to_csv("./test_save/" + path_ + "ssim_save/epoch" + str(epoch) + ".csv", index=False, sep=',')
        dataframe = pd.DataFrame({'psnr_value': psnr_list})
        dataframe.to_csv("./test_save/" + path_ + "psnr_save/epoch" + str(epoch) + ".csv", index=False, sep=',')
        print("The psnr&ssim&loss average value is:")
        print(np.mean(psnr_list), np.mean(ssim_list))





if __name__ == "__main__":
    dncnn = DnCNN()
    paths = ["Set9/", "BSD68/"]
    for i in range(2):
        path_ = paths[i]
        if path_ == "Set9/":
            for j in range(20):
                dncnn.img_test(path_, j)
        # else:
        #     dncnn.img_test(path_, 0)


