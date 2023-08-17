require('dotenv').config();

const Path = require('path')
    , FS = require('fs')
    , _ = require('lodash')
    , assert = require('assert')
;

const {
  Zod,
  Prompt,
  DocumentPrompt,
  TextSplitter,
} = require('../lib/llm.js');

describe('LLM tests', function() {
  this.timeout(0);

  it('returns JSON from prompt', async function() {
    let count = 5,
        things = 'ice cream flavors',
        schema = Zod.array(Zod.string());

    const prompt = new Prompt({
      messages: 'Return a list of {count} {things}'
    , model: 'gpt-3.5-turbo'
    , schema
    });

    let data = await prompt.call({
      count,
      things,
    });

    assert(data.response.length === count, 'Should have the correct number of items in the response');
    assert(schema.safeParse(data.response).success, 'Response should match the predefined schema');
    assert(data.cost > 0, 'Cost should be greater than 0');
    assert(data.tokensSent > 0, 'Tokens sent should be greater than 0');
    assert(data.tokensReceived > 0, 'Tokens received should be greater than 0');
  });

  it('uses templating in prompt messages', async function() {
    let count = 5,
        things = 'ice cream flavors',
        schema = Zod.array(Zod.string());

    const prompt = new Prompt({
      messages: [
        {role: 'system', content: 'Write responses in French'},
        'Return a list of {count} <%= things %>'
      ]
    , model: 'gpt-3.5-turbo'
    , schema
    });

    let data = await prompt.call({
      count,
      things,
    });

    assert(data.response.length === count, 'Should have the correct number of items in the response');
    assert(schema.safeParse(data.response).success, 'Response should match the predefined schema');
    assert(data.cost > 0, 'Cost should be greater than 0');
    assert(data.tokensSent > 0, 'Tokens sent should be greater than 0');
    assert(data.tokensReceived > 0, 'Tokens received should be greater than 0');
  });

  let text = FS.readFileSync(Path.join(__dirname, './data/shakespeare.txt')).toString();

  it('splits text', async function() {
    let model = 'gpt-3.5-turbo';

    const splitter = new TextSplitter({
      chunkSize: 100
    , chunkOverlap: 50
    , model
    });

    let {totalTextTokenLength, excerpts} = await splitter.splitText(text);

    let tokenCount;

    // Confirm totalTextTokenLength > 0
    assert(totalTextTokenLength > 0, 'Total text token length should be greater than 0');

    assert(excerpts.length > 0, 'Returned some excerpts');

    // Confirm each excerpt has a key excerpt with a string
    excerpts.forEach(excerpt => {
      assert(typeof excerpt.excerpt === 'string', 'Each excerpt should have a key "excerpt" with a string');
    });
  });

  it('summarizes document', async function() {
    let model = 'gpt-3.5-turbo';
    let schema = Zod.string().describe('summarization of speech');

    let prompt = new DocumentPrompt({
      responseTokenLength: 10,
      documentDescription: 'a motivational speech',
      instructions: 'summarize this speech into 10 words that give the main idea',
      schema,
      model,
    });

    let data = await prompt.call(text);

    assert(data.response.length, 'Should have a response');
    assert(schema.safeParse(data.response).success, 'Response should match the predefined schema');
    assert(data.cost > 0, 'Cost should be greater than 0');
    assert(data.tokensSent > 0, 'Tokens sent should be greater than 0');
    assert(data.tokensReceived > 0, 'Tokens received should be greater than 0');
  });

});
