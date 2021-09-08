
from transformers.models.bert.modeling_bert import *

class MyBertModel(BertModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)

    def forward(self, bert_indices, bert_segments, tune_start_layer):
        attention_mask = torch.ones_like(bert_indices)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        if tune_start_layer == 0:
            embedding_output = self.embeddings(bert_indices, token_type_ids=bert_segments)
            last_output, encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
            return encoder_outputs
        else:
            with torch.no_grad():
                embedding_output = self.embeddings(bert_indices, token_type_ids=bert_segments)
                all_hidden_states = (embedding_output,)
                for i in range(tune_start_layer-1):
                    layer_module = self.encoder.layer[i]
                    layer_outputs = layer_module(embedding_output, extended_attention_mask, head_mask[i])

                    embedding_output = layer_outputs[0]
                    all_hidden_states = all_hidden_states + (embedding_output,)

            for i in range(tune_start_layer-1, self.config.num_hidden_layers):
                layer_module = self.encoder.layer[i]
                layer_outputs = layer_module(embedding_output, extended_attention_mask, head_mask[i])

                embedding_output = layer_outputs[0]
                all_hidden_states = all_hidden_states + (embedding_output,)

            return all_hidden_states


class BertExtractor(nn.Module):
    def __init__(self, config):
        super(BertExtractor, self).__init__()
        self.config = config
        # self.bert = MyBertModel.from_pretrained(config.bert_path, force_download=True)
        self.bert = MyBertModel.from_pretrained(config.bert_path)
        self.bert.encoder.output_hidden_states = config.output_hidden_states
        self.bert.encoder.output_attentions = config.output_attentions
        self.bert_hidden_size = self.bert.config.hidden_size
        self.bert_layers = self.bert.config.num_hidden_layers + 1
        self.tune_start_layer = config.tune_start_layer


        for p in self.bert.named_parameters():
            # print(p[0])
            p[1].requires_grad = False
        for p in self.bert.named_parameters():
            items = p[0].split('.')
            if len(items) < 3: continue
            if items[0] == 'embeddings' and 0 >= self.tune_start_layer: p[1].requires_grad = True
            if items[0] == 'encoder' and items[1] == 'layer':
                layer_id = int(items[2]) + 1
                if layer_id >= self.tune_start_layer: p[1].requires_grad = True

    def forward(self, bert_indices, bert_segments, bert_pieces):
        all_outputs = self.bert(bert_indices, bert_segments, self.tune_start_layer)
        # outputs = []
        # or idx in range(self.bert_layers):
            # cur_output = torch.bmm(bert_pieces, all_outputs[idx])
            # outputs.append(cur_output)
        # return outputs
        output = torch.bmm(bert_pieces, all_outputs[self.bert_layers - 1])
        return output
