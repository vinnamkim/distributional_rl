def smooth_update(policy_net, target_net, tau):
    model_policy_net = policy_net.state_dict()
    model_target_net = target_net.state_dict()
    for param in model_target_net.keys():
        model_target_net[param] = (1-tau) * model_target_net[param] + tau * model_policy_net[param]
    target_net.load_state_dict(model_target_net)
    return

