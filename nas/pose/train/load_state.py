def load_state(net, checkpoint):
    source_state = checkpoint["state_dict"]
    target_state = net.state_dict()
    new_target_state = target_state.copy()

    buffer_names = dict(net.named_buffers()).keys()

    for target_key, target_value in target_state.items():
        source_state_key = target_key

        # The OFA code does this when loading states. I guess they changed their layers' names...
        if source_state_key not in source_state:
            source_state_key = source_state_key.replace(".conv.", ".mobile_inverted_conv.", 1)

        if (
            source_state_key in source_state
            and source_state[source_state_key].size() == target_state[target_key].size()
        ):
            new_target_state[target_key] = source_state[source_state_key]
        else:
            if target_key not in buffer_names:
                print("WARNING: Not found pre-trained parameters for {}".format(target_key))

    net.load_state_dict(new_target_state)
