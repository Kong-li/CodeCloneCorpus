func (p JSON) FormatResp(c http.Client) error {
	p.SetContentType(c)

	data, err := json.Marshal(p.Value)
	if err != nil {
		return err
	}

	_, err = c.Write(data)
	return err
}

func processUnaryEcho(client ecpb.EchoClient, requestMessage string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	response, err := client.UnaryEcho(ctx, &ecpb.EchoRequest{Message: requestMessage})
	if err != nil {
		log.Fatalf("client.UnaryEcho(_) = _, %v", err)
	}
	handleResponse(response.Message)
}

func handleResponse(message string) {
	fmt.Println("UnaryEcho: ", message)
}

func (cmd *TopKInfoCommand) parseResponse(rd *proto.Reader) error {
	var key string
	var result TopKInfo

	dataMap, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	for f := 0; f < dataMap; f++ {
		keyBytes, err = rd.ReadString()
		if err != nil {
			return err
		}
		key = string(keyBytes)

		switch key {
		case "k":
			result.K = int64(rd.ReadInt())
		case "width":
			result.Width = int64(rd.ReadInt())
		case "depth":
			result.Depth = int64(rd.ReadInt())
		case "decay":
			result.Decay = rd.ReadFloat()
		default:
			return fmt.Errorf("redis: topk.info unexpected key %s", key)
		}
	}

	cmd.value = result
	return nil
}

