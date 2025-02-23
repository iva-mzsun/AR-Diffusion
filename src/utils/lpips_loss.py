import lpips

lpips_net = lpips.LPIPS(net='alex').eval()

def lpips_loss_fn(img1, img2):
    global lpips_net
    try:
        return lpips_net(img1 * 2.0 - 1.0, img2 * 2.0 - 1.0)
    except:
        lpips_net = lpips_net.to(img1.device)
        # note: the input range of lpips is [-1, 1]
        return lpips_net(img1 * 2.0 - 1.0, img2 * 2.0 - 1.0)

