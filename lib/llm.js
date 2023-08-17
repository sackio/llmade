require('dotenv').config();

const _ = require('lodash');
const Delay = require('delay');

const { zodToJsonSchema } = require('zod-to-json-schema');

const { 
  ChatOpenAI
} = require('langchain/chat_models/openai');

const {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  AIMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
  PipelinePromptTemplate,
} = require('langchain/prompts');

const { 
  StructuredOutputParser,
  OutputFixingParser,
} = require('langchain/output_parsers');

const { LLMChain } = require('langchain/chains');

const {
  SupportedTextSplitterLanguages,
  RecursiveCharacterTextSplitter,
  TokenTextSplitter,
} = require('langchain/text_splitter');

const Z = require('zod');

const { RateLimiter } = require('./limiter.js');

const modelSettings = require('../settings.json').models.filter(m => m.modelName.match(/^gpt/i));

const rateLimiters = modelSettings.reduce((acc, settings) => {
  acc[settings.modelName] = new RateLimiter(settings);
  return acc;
}, {});

class Model {
  constructor(settings = {}) {
    // Extract settings
    const { modelName, maxTokens, tokenTxPrice, tokensRxPrice, requestsPerMinute, tokensPerMinute, ...modelSettings } = settings;

    // Assign settings
    this.modelName = modelName;
    this.maxTokens = maxTokens;
    this.tokenTxPrice = tokenTxPrice;
    this.tokensRxPrice = tokensRxPrice;
    this.requestsPerMinute = requestsPerMinute;
    this.tokensPerMinute = tokensPerMinute;

    this.rateLimiter = rateLimiters[this.modelName]; //shared rate limiters
    this.llm = new ChatOpenAI(Object.assign({}, { modelName, ...modelSettings }));
  }

  calculateCost(tokensSent, tokensReceived) {
    const cost = (tokensSent * this.tokenTxPrice) + (this.tokensRxPrice * tokensReceived);
    return cost;
    //return parseFloat(cost.toFixed(2));
  }

  async countTokens(text, debug) {
    if (debug) console.log(text);
    return await this.llm.getNumTokens(text);
  }

  async remainingTokensForPrompt(prompt) {
    const tokenCount = await this.countTokens(prompt.toString());
    return this.maxTokens - tokenCount;
  }
}

const models = modelSettings.reduce((acc, settings) => {
  acc[settings.modelName] = new Model(settings);
  return acc;
}, {});

class JSONParser {
  constructor(schema, modelName) {
    this.zod = schema;
    this.schema = zodToJsonSchema(schema);
    this.description = JSON.stringify(this.schema);

    this.parser = StructuredOutputParser.fromZodSchema(schema);
    this.formatInstructions = this.parser.getFormatInstructions();

    this.model = new Model({
      modelName
    , temperature: 0
    });
    this.fixingParser = OutputFixingParser.fromLLM(
      this.model.llm,
      this.parser
    );
  }

  async parse(text, rateLimiter, promptTokens=0) {
    let res = {};
    try {
      res.data = await this.parser.parse(text);
    } catch (e) {    
      const self = this;
      res = await this.model.rateLimiter.process(async (reportTokens) => {
        const data = await this.fixingParser.parse(text)
            , tokensSent = Math.round(promptTokens * 1.10)
            , tokensReceived = await self.model.countTokens(JSON.stringify(data))
            , cost = self.model.calculateCost(tokensSent, tokensReceived);

        reportTokens(tokensSent, tokensReceived);

        return {
          data,
          tokensSent,
          tokensReceived,
          cost,
        };
      });

      res.parserFixed = true;
    }

    return res;
  }  
}

class Prompt {
  constructor({
    messages
  , schema
  , model='gpt-4'
  , modelSettings={}
  , inputVariables
  , messageData={}
  }) {
    this.setMessages(messages, messageData);

    this.schema = schema;

    this.setModel(model, modelSettings);
  }

  setMessages(messages, data={}, modelSettings) {
    this.messageText = messages;
    if (!Array.isArray(this.messageText)) this.messageText = [this.messageText];

    this.messages = this.messageText.map(message => {
      let content = typeof message === 'string' ? message : message.text || message.content

      try {
        content = _.template(content)(data);
      } catch {

      }

      if (typeof message === 'string') {
        return HumanMessagePromptTemplate.fromTemplate(content);
      } else if (message.role === 'user' || message.role === 'human') {
        return HumanMessagePromptTemplate.fromTemplate(content);
      } else if (message.role === 'ai' || message.role === 'assistant') {
        return AIMessagePromptTemplate.fromTemplate(content);        
      } else if (message.role === 'system') {
        return SystemMessagePromptTemplate.fromTemplate(content);
      } else {
        throw new Error('Invalid message type');
      }
    });

    this.promptTemplate = ChatPromptTemplate.fromPromptMessages(this.messages);

    if (this.modelName) this.setModel(this.modelName, modelSettings);
  }

  setModel(model, newModelSettings={}) {
    this.modelName = model;
    if (_.size(newModelSettings)) this.modelSettings = _.extend({}, modelSettings.find(s => s.modelName === model), newModelSettings);

    this.model = _.size(this.modelSettings) ? new Model({
      modelName: this.modelName,
      ...this.modelSettings
    }) : models[this.modelName];
    this.chain = new LLMChain({
      llm: this.model.llm,
      prompt: this.promptTemplate
    });

    this.parser = new JSONParser(this.schema, this.model.modelName);
    this.formatInstructions = this.parser.formatInstructions;
  }

  async call(data, retries=5, retryDelay=1000) {
    let tokensSent = 0
      , tokensReceived = 0
      , cost = 0;

    let promptData = this.promptData(data)
      , tokenCount = await this.countTokens(promptData)
      , remainingTokenCount = this.model.maxTokens - tokenCount;

    if (remainingTokenCount < 0) {
      retries = 0;
      throw new Error('tokenCount exceeds maxTokens');
    }

    promptData = _.extend({}, this, {
      remainingTokenCount
    , tokenCount
    }, data);

    this.setMessages(this.messageText, promptData);

    if (data.dryrun) {
      tokensSent += await this.countTokens(promptData);
      tokensReceived += promptData.responseTokenLength || (this.model.maxTokens - tokensSent)
      cost += this.model.calculateCost(tokensSent, tokensReceived);
      return {
        tokensSent,
        tokensReceived,
        cost,
      };
    }

    while (retries > 0) {
      try {
        let response = await this.model.rateLimiter.process(async (reportTokens) => {
          const rawResponse = await this.chain.call(promptData);

          tokensSent += await this.countTokens(promptData);
          tokensReceived += await this.model.countTokens(rawResponse.text);
          cost += this.model.calculateCost(tokensSent, tokensReceived);

          reportTokens(tokensSent + tokensReceived);

          const parsedResponse = await this.parser.parse(rawResponse.text, this.model.rateLimiter, tokensSent);

          tokensSent += (parsedResponse.tokensSent || 0);
          tokensReceived += (parsedResponse.tokensReceived || 0);
          cost += (parsedResponse.cost || 0);

          return {
            response: parsedResponse.data
          , tokensSent
          , tokensReceived
          , cost
          };
        });

        return response;
      } catch (error) {
        retries--;
        //console.log(`An error occurred: ${error.message}. Retrying... Remaining attempts: ${retries}`);
        if (retries === 0) throw error; // if all retries have been used, rethrow the error
        await Delay(retryDelay);        
      }
    }
  }

  promptData(data={}, fillString='xxx') {
    let promptData = _.extend(this.promptTemplate.inputVariables.reduce((obj, str) => ({ ...obj, [str]: fillString }), {}), this, data);
    this.setMessages(this.messageText, promptData);
    promptData = _.extend(this.promptTemplate.inputVariables.reduce((obj, str) => ({ ...obj, [str]: fillString }), {}), this, data);
    return promptData;
  }

  async countTokens(data, debug, fillString) {
    let promptData = this.promptData(data, fillString);
    const prompt = await this.chain.prompt.formatPromptValue(promptData);
    const tokenCount = await this.model.countTokens(_.map(prompt.messages, 'text').join('\n'), debug);
    return tokenCount;
  }

  async countRemainingTokens(data, debug, fillString='xxx') {
    const tokenCount = await this.countTokens(data, debug, fillString)
    const remainingTokenCount = this.model.maxTokens - tokenCount;
    return remainingTokenCount;
  }

  async chunksFromArray(arr='', maxTokens, delimeter='\n') {
    let chunks = []
      , curChunk
      , tokenCount = 0
      , delimeterTokens = await this.model.countTokens(delimeter);

    for (let i = 0; i < arr.length; i++) {
      const el = arr[i];
      if (!el) continue;

      let tokens = await this.model.countTokens(el) + (curChunk ? delimeterTokens : 0);

      if (!curChunk || ((tokens + tokenCount) > maxTokens)) {
        if (curChunk) {
          chunks.push(curChunk);
        }

        curChunk = '';
        tokenCount = 0;
        tokens -= delimeterTokens;
      }

      //if (tokens > maxTokens) throw new Error('chunk exceeds max tokens');

      curChunk = curChunk ? `${curChunk}${delimeter}${el}` : el;
      tokenCount += tokens;
    }

    if (curChunk) chunks.push(curChunk);

    return chunks;
  }
}

class TextSplitter {
  constructor({
    chunkSize
  , chunkOverlap
  , bufferPercentage
  , model
  , splitter=TokenTextSplitter
  }) {
    this.setSplitter({
      chunkSize
    , chunkOverlap
    , bufferPercentage
    , splitter
    });

    this.model = typeof model !== 'string' ? model : models[model];
  }

  setSplitter({
    chunkSize,
    chunkOverlap=0,
    bufferPercentage=1.0,
    splitter=TokenTextSplitter
  }) {
    this.chunkSize = Math.floor(chunkSize * bufferPercentage);
    this.chunkOverlap = chunkOverlap;

    this.splitter = new splitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap
    });
  }

  async splitText(text, debug) {
    let totalTextTokenLength = await this.model.countTokens(text, debug);

    let chunks = await this.splitter.createDocuments([text]);
    chunks = _.filter(_.map(chunks, 'pageContent'), c => c);

    if (!chunks?.length) chunks = [text];

    totalTextTokenLength += this.splitter.chunkOverlap * chunks.length;

    const excerpts = [];
    for (const excerpt of chunks) {
      const excerptTokenLength = await this.model.countTokens(excerpt);

      excerpts.push({
        excerpt,
        excerptTokenLength,
        excerptPercentageLength: Math.round(excerptTokenLength / totalTextTokenLength * 100)
      });
    }

    return {
      totalTextTokenLength
    , excerpts
    };
  }
}

class DocumentPrompt {
  constructor({
    model,
    responseTokenLength,
    schema,
    systemInstructions,
    instructions,
    documentDescription,
  }) {
    this.model = model;
    this.schema = schema;

    this.systemInstructions = systemInstructions;
    this.instructions = instructions;
    this.documentDescription = documentDescription;

    this.setResponseTokenLength(responseTokenLength);

    this.prompt = new Prompt({
      messages: [
        {
          role: 'system'
        , content: `You are a generative LLM with a maximum token length of {maxTokens}. You are designed to scan over a long amount of text in excerpts, returning a response based on the user's instructions. As you see more excerpts, you may update your response, or leave it unchanged if new excerpts contain no new relevant information for your task.

<% if (typeof systemInstructions === 'undefined') { %>
{systemInstructions}

<% } %>
{formatInstructions}`
        },
        `You are generating a {responseTokenLength} token response based on {documentDescription} that is {totalTextTokenLength} tokens long in total. 

<% if (typeof response === 'undefined') { %>
This is the first excerpt, with {excerptTokenLength} tokens of text. It represents the first {excerptPercentageLength}% of the text:
<% } else { %>
So far you have seen {currentPercentageLength}% of the text, and you still have not seen the final {remainingPercentageLength}% of the text.

Your response so far (based on the first {currentPercentageLength}% of the text):
=========
{response}
=========

This is an excerpt with the next {excerptTokenLength} tokens of text. It represents the next {excerptPercentageLength}% of text:
<% } %>
---------
{excerpt}
---------

{instructions}`
      ],
      model: this.model,
      schema: this.schema,
    });

    this.maxTokens = this.prompt.model.maxTokens;    
  }

  setResponseTokenLength(responseTokenLength) {
    this.responseTokenLength = responseTokenLength;
  }

  async setTextSplitter({
    chunkSize,
    chunkOverlap=0,
    bufferPercentage=0.95,
  }) {
    this.remainingTokens = await this.prompt.countRemainingTokens({}, false, '999999');
    this.remainingTokens -= Math.ceil(this.responseTokenLength * 2);
    this.remainingTokens = Math.floor(this.remainingTokens * bufferPercentage);

    this.chunkSize = chunkSize || this.remainingTokens;
    this.chunkOverlap = chunkOverlap;

    this.splitter = new TextSplitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
      model: this.prompt.model
    });
  }

  async call(text, settings={}) {
    if (!this.splitter) await this.setTextSplitter(settings);

    let {
      excerpts,
      totalTextTokenLength,
    } = await this.splitter.splitText(text);

    let count = 0
      , tokensSent = 0
      , tokensReceived = 0
      , cost = 0
      , response
      , currentTokenCount = 0;

    for (const excerpt of excerpts) {
      try {
        const currentPercentageLength = Math.round(currentTokenCount / totalTextTokenLength * 100);

        const args = _.extend({}, _.omit(this, [
          'prompt'
        , 'schema'
        , 'splitter'
        ]), settings, excerpt, {
          totalTextTokenLength,
          currentTokenCount,
          currentPercentageLength,
          remainingPercentageLength: 100 - excerpt.excerptPercentageLength - currentPercentageLength,
          response: response ? JSON.stringify(response) : undefined,
        });

        if (settings.dryrun) {
          const tokenCount = await this.prompt.countTokens(args);
          tokensSent += tokenCount + this.responseTokenLength;
          tokensReceived += tokenCount + this.responseTokenLength;
          cost += this.prompt.model.calculateCost(tokensSent, tokensReceived);
        } else {
          const data = await this.prompt.call(args);
          cost += data?.cost || 0;
          tokensSent += (data?.tokensSent || 0);
          tokensReceived += (data?.tokensReceived || 0);
          response = data?.response || response;
        }

        //cost = parseFloat(cost.toFixed(2));
        currentTokenCount += excerpt.excerptTokenLength;
        count++;

        if (settings?.progress) {
          let progressResponse = await settings.progress(_.extend({}, args, {
            cost
          , tokensSent
          , tokensReceived
          , response
          , count
          }));
          if (progressResponse === 'stop') break;
        }

      } catch (e) {
        console.log(e);
      }
    }

    return {
      response
    , cost
    , tokensSent
    , tokensReceived
    };
  }
}

module.exports = {
  JSONParser,
  Prompt,
  Zod: Z,
  TextSplitter, 
  DocumentPrompt,
  Model,
  models,
};