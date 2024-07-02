# null instruct jaについて
null instruct jaはある程度日本語性能が高いllmを用いて、人の手を極力介さずにinstructデータを作成するプロジェクトです。

# ライセンス(重要)
- このプロジェクトのコードを継承するコードおよびコードの実行によって得られた成果物は、オープンソースへの貢献のため、必ずオープンソースとして公開しなくてはなりません。  
- このプロジェクトのコードを継承するコードおよびコードの実行によって得られた成果物は、このライセンスと同等の意義を持つライセンスを継承しなくてはなりません。  
- 上二つの制限事項はtwitter(advictrius85)のDMにある程度納得できる理由を載せたうえで私が同意すれば撤廃可能(クローズにしてもよい)です。  
- このコードが世間でどのように使われたかを知りたいため、可能であれば、https://github.com/neodyland/null-instruct-ja のissueに成果物が公開されたurlを載せてくださるとありがたいです！  
- その他制約事項についてはMITライセンスに準じます。

# 実行
1. http://localhost:8080/completions にpostできる状態でgemma-2-9b-itまたはgemma-2-27b-itを起動します。
2. このリポジトリをクローンし、`requirements.txt`から依存関係をインストールします。
3. `API_HOST=http://localhost:8080 python gen_prompt.py --max_count_for_evol 3 --max_count_for_null 300 --evol_steps 1` を実行します。
4. 終わったら`python gen_response.py` を実行します。
5. 終わったら`python plot.py` を実行します。
6. `result/filtered.json` に結果が出力されます。

# オプションについて
## max_count_for_null
無からプロンプトを生成する回数を指定します。この回数は最終的にほしいプロンプトの量によって変動させてください。  
最終的に生成されるプロンプトは約max_count_for_null * max_count_for_evol / 5となります。
## max_count_for_evol
nullで生成されたpromptに対して類似するプロンプトを生成する回数を指定します。3回を強く推奨します。
## evol_steps
プロンプトの進化回数を指定します。gemma-2の場合は一回くらいで十分だと思います。

# それぞれのファイルの役割
## ds.py
```python
from ds import ds
ds.push_to_hub("<hf repo id>")
```
で素早くpushするためのファイルです。
## gen_prompt.py
無からプロンプトを生成するところまでをやってくれます。
## gen_response.py
プロンプトからいい感じの回答を生成します。
## gen.py
サーバーへのリクエストの共通事項をまとめてあります。
## plot.py
`plot_after.png`, `plot_before.png` (doc2vecによる文章分布の確認) および`result/filtered.json`(類似した質問の排除)を生成します。