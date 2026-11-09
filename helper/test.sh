for i in {0..4};do
time curl -X POST http://127.0.0.1:18005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.3-Flash-Mix",
    "max_tokens":1024,
    "messages": [
      {
        "role": "user",
        "content": "在一个已支持下述组合的实现中，部署 A 只开 PCP：两张 PCP 卡各有每条请求的完整历史副本，decode 按请求选择 owner。部署 B 改用 DCP，把每条请求的历史位置分到这两张卡上。工程师复用 A 的 owner 策略，让 B 每步只由一张卡计算，不取回另一张卡的 KV，只把本地序列长度设成全局长度。为什么这不能恢复正确的 attention？如果改为两张卡拿到相同查询、分别计算局部 attention，再按各自的有效 Token 数加权平均输出，是否已经修好？还需要什么合并信息，为什么？"
      }
    ],
    "stream": false,
    "temperature": 0
  }'
echo -e "\n========= Request $i done =========\n"
done