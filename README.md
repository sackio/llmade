# LLMADE

LLMADE is a refreshing library that offers tools to squeeze the most out of LLMs. LLMADE serves up a full glass of utilities for interacting with generative models.

## Installation

To get started with LLMADE, install the package using npm:

```bash
npm install llmade

```

## Example Usage

### Creating and Using Prompts

```javascript
const { Prompt, Zod } = require('llmade');

let count = 5,
    things = 'ice cream flavors',
    schema = Zod.array(Zod.string());

const prompt = new Prompt({
  messages: 'Return a list of {count} {things}',
  model: 'gpt-3.5-turbo',
  schema,
});

let data = await prompt.call({
  count,
  things,
});
```

### Templating in Prompt Messages

```javascript
const { Prompt, Zod } = require('llmade');

const prompt = new Prompt({
  messages: [
    {role: 'system', content: 'Write responses in French'},
    'Return a list of {count} <%= things %>'
  ],
  model: 'gpt-3.5-turbo',
  schema,
});

let data = await prompt.call({
  count: 5,
  things: 'ice cream flavors',
});
```

### Splitting Text into Excerpts

```javascript
const { TextSplitter } = require('llmade');
const FS = require('fs');
const Path = require('path');

let text = FS.readFileSync(Path.join(__dirname, './data/shakespeare.txt')).toString();

const splitter = new TextSplitter({
  chunkSize: 100,
  chunkOverlap: 50,
  model: 'gpt-3.5-turbo',
});

let {totalTextTokenLength, excerpts} = await splitter.splitText(text);
```

### Summarizing Documents

```javascript
const { DocumentPrompt, Zod } = require('llmade');

let schema = Zod.string().describe('summarization of speech');

let prompt = new DocumentPrompt({
  responseTokenLength: 10,
  documentDescription: 'a motivational speech',
  instructions: 'summarize this speech into 10 words that give the main idea',
  schema,
  model: 'gpt-3.5-turbo',
});

let data = await prompt.call(text);
```
