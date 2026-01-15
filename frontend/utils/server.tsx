import "server-only";
import { AIProvider } from "./client";
import { ReactNode } from "react";
import { Runnable } from "@langchain/core/runnables";
import { CompiledStateGraph } from "@langchain/langgraph";
import { createStreamableUI, createStreamableValue } from "ai/rsc";
import { StreamEvent } from "@langchain/core/tracers/log_stream";

export const LAMBDA_STREAM_WRAPPER_NAME = "lambda_stream_wrapper";

export type RunUICallbacks = Record<
  string,
  ReturnType<typeof createStreamableUI | typeof createStreamableValue>
>;
export type EventHandlerFields = {
  ui: ReturnType<typeof createStreamableUI>;
  callbacks: RunUICallbacks;
};
export type EventHandler =
  | ((event: StreamEvent, fields: EventHandlerFields) => void)
  | ((event: StreamEvent, fields: EventHandlerFields) => Promise<void>);

/**
 * Executes `streamEvents` method on a runnable
 * and converts the generator to a RSC friendly stream
 *
 * @param runnable
 * @returns React node which can be sent to the client
 */
export function streamRunnableUI<RunInput, RunOutput>(
  runnable:
    | Runnable<RunInput, RunOutput>
    | CompiledStateGraph<RunInput, Partial<RunInput>>,
  inputs: RunInput,
  options: {
    eventHandlers: Array<EventHandler>;
  },
) {
  const ui = createStreamableUI();
  const [lastEvent, resolve, reject] = withResolvers<
    Array<any> | Record<string, any>
  >();
  
  // Prevent unhandled rejection on the server since this promise is passed to client
  lastEvent.catch(() => {});

  let shouldRecordLastEvent = true;

  (async () => {
    let lastEventValue: StreamEvent | null = null;
    const callbacks: RunUICallbacks = {};

    try {
      for await (const streamEvent of (
        runnable as Runnable<RunInput, RunOutput>
      ).streamEvents(inputs, {
        version: "v1",
      })) {
        for await (const handler of options.eventHandlers) {
          await handler(streamEvent, {
            ui,
            callbacks,
          });
        }
        if (shouldRecordLastEvent) {
          lastEventValue = streamEvent;
        }
        if (
          streamEvent.data.chunk?.name === "LangGraph" &&
          streamEvent.data.chunk?.event === "on_chain_end"
        ) {
          shouldRecordLastEvent = false;
        }
      }

      // resolve the promise, which will be sent
      // to the client thanks to RSC
      const resolveValue =
        lastEventValue?.data.output || lastEventValue?.data.chunk?.data?.output;
      resolve(resolveValue);
    } catch (error) {
      // Handle connection errors, timeouts, etc.
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      console.error("[streamRunnableUI] Stream error:", errorMessage);

      // Reject the promise so callers can handle it
      reject(error);

      // Append error message to UI so user sees feedback
      ui.append(
        <div className="text-red-500 p-2 rounded bg-red-50 dark:bg-red-900/20">
          Connection error: Unable to reach the backend. Please ensure the server is running.
        </div>
      );
    } finally {
      // Always close all callbacks - some may already be closed by event handlers
      Object.values(callbacks).forEach((cb) => {
        try {
          cb.done();
        } catch {
          // Stream already closed by event handler
        }
      });

      // Always close the main UI stream
      try {
        ui.done();
      } catch {
        // Stream already closed
      }
    }
  })();

  return { ui: ui.value, lastEvent };
}

/**
 * Polyfill to emulate the upcoming Promise.withResolvers
 */
export function withResolvers<T>() {
  let resolve: (value: T) => void;
  let reject: (reason?: any) => void;

  const innerPromise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  // @ts-expect-error
  return [innerPromise, resolve, reject] as const;
}

/**
 * Expose these endpoints outside for the client
 * We wrap the functions in order to properly resolve importing
 * client components.
 *
 * TODO: replace with createAI instead, even though that
 * implicitly handles state management
 *
 * See https://github.com/vercel/next.js/pull/59615
 * @param actions
 */
export function exposeEndpoints<T extends Record<string, unknown>>(
  actions: T,
): {
  (props: { children: ReactNode }): Promise<JSX.Element>;
  $$types?: T;
} {
  return async function AI(props: { children: ReactNode }) {
    return <AIProvider actions={actions}>{props.children}</AIProvider>;
  };
}
